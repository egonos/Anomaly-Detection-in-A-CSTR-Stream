import pandas as pd
import numpy as np


def compute_outputs(T, Ca0, Q, output_noise_scale=0.07):
    # define constants
    R = 8.314  # heat constant: J mol^-1 K^-1
    A = 1e10  # Arrhenius constant: s^-1
    Ea = 80_000  # activation energy: J mol^-1
    V = 100  # volume: L

    k = A * np.exp(-Ea / (R * T))  # rate constant
    tau = V / Q  # space time
    X_A = (k * tau) / (1 + k * tau)
    X_A = np.clip(X_A, 0, 1)
    Ca = Ca0 * (1 - X_A)
    ra = -k * Ca

    # add zero mean Gaussian noise to make data more realistic
    Ca_measured = Ca + np.random.normal(0, output_noise_scale, len(Ca))
    ra_measured = ra + np.random.normal(0, output_noise_scale / 5, len(ra))
    Xa_measured = X_A + np.random.normal(0, output_noise_scale, len(X_A))

    return Xa_measured, Ca_measured, ra_measured


def generate_normal_samples(n_samples, input_noise_scale=0.03):

    T = np.random.normal(350, 3, n_samples) + np.random.normal(
        0, input_noise_scale * 350, n_samples
    )
    Ca0 = np.random.normal(1, 0.02, n_samples) + np.random.normal(
        0, input_noise_scale, n_samples
    )
    Q = np.random.uniform(1, 1.5, n_samples) + np.random.normal(
        0, input_noise_scale, n_samples
    )

    X_A, Ca, ra = compute_outputs(T, Ca0, Q)

    df = pd.DataFrame(
        {
            "temperature": np.round(T, 2),
            "Ca0": np.round(Ca0, 4),
            "Q": np.round(Q, 3),
            "Xa": np.round(X_A, 4),
            "ra": np.round(ra, 5),
            "Ca": np.round(Ca, 4),
            "is_anomaly": np.zeros(n_samples, dtype=int),
            "anomaly_type": np.array(["none"] * n_samples, dtype=object),
        }
    )

    return df


def generate_abnormal_samples(n_samples, input_noise_scale=0.03):
    per_type = n_samples // 6
    actual_samples = per_type * 6
    print("Samples rounded to:", actual_samples)

    # define anomalies
    T_low = np.random.normal(300, 3, per_type) + np.random.normal(
        0, input_noise_scale * 350, per_type
    )
    T_high = np.random.normal(400, 3, per_type) + np.random.normal(
        0, input_noise_scale * 350, per_type
    )
    T_norm = np.random.normal(350, 3, per_type) + np.random.normal(
        0, input_noise_scale * 350, per_type
    )

    Ca0_low = np.random.normal(0.01, 0.02, per_type) + np.random.normal(
        0, input_noise_scale, per_type
    )
    Ca0_high = np.random.normal(2, 0.02, per_type) + np.random.normal(
        0, input_noise_scale, per_type
    )
    Ca0_norm = np.random.normal(1.0, 0.02, per_type) + np.random.normal(
        0, input_noise_scale, per_type
    )

    Q_low = np.random.uniform(0.2, 0.5, per_type) + np.random.normal(
        0, input_noise_scale, per_type
    )
    Q_high = np.random.uniform(2, 2.5, per_type) + np.random.normal(
        0, input_noise_scale, per_type
    )
    Q_norm = np.random.uniform(1.0, 1.5, per_type) + np.random.normal(
        0, input_noise_scale, per_type
    )

    # combine anomalies
    T = np.concatenate([T_low, T_high, T_norm, T_norm, T_norm, T_norm])
    Ca0 = np.concatenate([Ca0_norm, Ca0_norm, Ca0_low, Ca0_high, Ca0_norm, Ca0_norm])
    Q = np.concatenate([Q_norm, Q_norm, Q_norm, Q_norm, Q_low, Q_high])

    Xa, Ca, ra = compute_outputs(T, Ca0, Q)

    anomaly_type = (
        ["T_low"] * per_type
        + ["T_high"] * per_type
        + ["Ca0_low"] * per_type
        + ["Ca0_high"] * per_type
        + ["Q_low"] * per_type
        + ["Q_high"] * per_type
    )

    df = pd.DataFrame(
        {
            "temperature": np.round(T, 2),
            "Ca0": np.round(Ca0, 4),
            "Q": np.round(Q, 3),
            "Xa": np.round(Xa, 4),
            "ra": np.round(ra, 5),
            "Ca": np.round(Ca, 4),
            "is_anomaly": np.ones(actual_samples, dtype=int),
            "anomaly_type": anomaly_type,
        }
    )

    return df


def error_summary(true_labels, preds, error_type):
    results = pd.DataFrame(
        {
            "True Labels": true_labels,
            "Predicted Labels": preds,
            "Error Type": error_type,
        }
    )
    results["matching_results"] = (
        results["True Labels"] == results["Predicted Labels"]
    ).astype(int)
    summary = pd.concat(
        [
            results.groupby("Error Type")["matching_results"].sum(),
            results.groupby("Error Type")["matching_results"].count(),
            results.groupby("Error Type")["matching_results"].mean() * 100,
        ],
        axis=1,
    )
    summary.columns = ["Matching Samples", "Total Samples", "Matching Rate"]
    return summary.round(2)
