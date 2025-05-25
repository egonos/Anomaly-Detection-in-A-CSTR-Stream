from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.log.logging_mixin import LoggingMixin


from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os
import joblib
import json
from scipy.stats import kstest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import generate_abnormal_samples, generate_normal_samples

N_SAMPLES = 1000
ABNORMAL_RATIO = 0.3
logger = LoggingMixin().log


def generate_reactor_data(n_samples=N_SAMPLES, abnormal_ratio=ABNORMAL_RATIO, **kwargs):

    n_normal = int(n_samples * (1 - abnormal_ratio))
    n_abnormal = n_samples - n_normal

    normal_samples = generate_normal_samples(n_samples=n_normal)
    abnormal_samples = generate_abnormal_samples(n_samples=n_abnormal)

    sensor_data = (
        pd.concat([normal_samples, abnormal_samples], axis=0, ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
        .iloc[:, :-2]
    )

    kwargs["ti"].xcom_push(
        key="sensor_data", value=sensor_data.to_dict(orient="records")
    )
    kwargs["ti"].xcom_push(key="column_names", value=sensor_data.columns.tolist())
    logger.info(
        f"Successfully generated {n_samples} samples with {abnormal_ratio*100}% abnormal samples."
    )


def find_anomalies(**kwargs):

    # pull the sensor data and column names from XCom
    sensor_data = kwargs["ti"].xcom_pull(
        key="sensor_data", task_ids="generate_reactor_data"
    )
    cols = kwargs["ti"].xcom_pull(key="column_names", task_ids="generate_reactor_data")
    df = pd.DataFrame(sensor_data, columns=cols)

    # get the model, the decision rule and the scaler
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    DECISION_RULE_PATH = os.path.join(MODEL_DIR, "decision_rule.json")

    iso = joblib.load(os.path.join(MODEL_DIR, "if_model.joblib"))
    ss = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    with open(DECISION_RULE_PATH, "r") as f:
        decision_rule = json.load(f)

    Q_MIN, Q_MAX = decision_rule["Q_MIN"], decision_rule["Q_MAX"]
    mask = (df["Q"] < Q_MIN) | (df["Q"] > Q_MAX)
    filtered_sensor_data = df[~mask]
    filtered_sensor_data = filtered_sensor_data.loc[
        :, filtered_sensor_data.columns != "Q"
    ]
    scaled_sensor_data = ss.transform(filtered_sensor_data)
    preds = np.where(iso.predict(scaled_sensor_data) == 1, 0, 1)

    final_preds = np.zeros(len(df), dtype=int)
    final_preds[mask.values] = 1
    final_preds[~mask.values] = preds

    kwargs["ti"].xcom_push(key="anomalies", value=final_preds.tolist())
    logger.info(
        f"Successfully found {len(df) - len(df[final_preds == 0])} anomalies in the data."
    )


def report_summary_stats(**kwargs):

    # pull the sensor data, column names and anomaly index from XCom
    sensor_data = kwargs["ti"].xcom_pull(
        key="sensor_data", task_ids="generate_reactor_data"
    )
    cols = kwargs["ti"].xcom_pull(key="column_names", task_ids="generate_reactor_data")
    anomalies = kwargs["ti"].xcom_pull(key="anomalies", task_ids="find_anomalies")

    df = pd.DataFrame(sensor_data, columns=cols)
    filtered_df = df[np.array(anomalies) == 0]

    # Calculate summary statistics
    mean_temp, std_temp = (
        filtered_df["temperature"].mean(),
        filtered_df["temperature"].std(),
    )
    min_temp, max_temp = (
        filtered_df["temperature"].min(),
        filtered_df["temperature"].max(),
    )
    temp_pval = kstest(filtered_df["temperature"], "norm")[1]

    mean_Ca0, std_Ca0 = filtered_df["Ca0"].mean(), filtered_df["Ca0"].std()
    min_Ca0, max_Ca0 = filtered_df["Ca0"].min(), filtered_df["Ca0"].max()
    Ca0_pval = kstest(filtered_df["Ca0"], "norm")[1]

    min_Q, max_Q = filtered_df["Q"].min(), filtered_df["Q"].max()
    number_anomalies = len(df) - len(filtered_df)

    #
    hook = PostgresHook(postgres_conn_id="sensor_records")
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reactor_data (
            id SERIAL PRIMARY KEY,
            mean_temperature FLOAT,
            std_temperature FLOAT,
            min_temperature FLOAT,
            max_temperature FLOAT,
            temperature_ks_pval FLOAT,
            mean_Ca0 FLOAT,
            std_Ca0 FLOAT,
            min_Ca0 FLOAT,
            max_Ca0 FLOAT,
            Ca0_ks_pval FLOAT,
            min_Q FLOAT,
            max_Q FLOAT,
            number_anomalies INT);
        """
    )

    cursor.execute(
        """
        INSERT INTO reactor_data
        (mean_temperature, std_temperature, min_temperature, max_temperature,
        temperature_ks_pval, mean_Ca0, std_Ca0, min_Ca0, max_Ca0,
        Ca0_ks_pval, min_Q, max_Q, number_anomalies)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            float(mean_temp),
            float(std_temp),
            float(min_temp),
            float(max_temp),
            float(temp_pval),
            float(mean_Ca0),
            float(std_Ca0),
            float(min_Ca0),
            float(max_Ca0),
            float(Ca0_pval),
            float(min_Q),
            float(max_Q),
            int(number_anomalies),
        ),
    )

    conn.commit()
    cursor.close()
    conn.close()

    logger.info("Successfully reported summary statistics to the database.")


with DAG(
    dag_id="anomaly_detection_dag",
    start_date=datetime(2025, 1, 1),
    schedule="@hourly",
    catchup=False,
) as dag:

    generate_reactor_data_task = PythonOperator(
        task_id="generate_reactor_data",
        python_callable=generate_reactor_data,
        provide_context=True,
        op_kwargs={"n_samples": N_SAMPLES, "abnormal_ratio": ABNORMAL_RATIO},
    )

    find_anomalies_task = PythonOperator(
        task_id="find_anomalies", python_callable=find_anomalies, provide_context=True
    )

    report_summary_stats_task = PythonOperator(
        task_id="report_summary_stats",
        python_callable=report_summary_stats,
        provide_context=True,
    )

    generate_reactor_data_task >> find_anomalies_task >> report_summary_stats_task
