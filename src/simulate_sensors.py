import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_samples: int = 1000,
    fault_ratio: float = 0.15,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic sensor data for a railway electrical system.

    Features:
    - line_voltage_kV
    - line_current_A
    - transformer_temp_C
    - vibration_g
    - power_factor
    - frequency_Hz
    - fault (0 = normal, 1 = fault)
    """
    rng = np.random.RandomState(random_state)

    n_faults = int(n_samples * fault_ratio)
    n_normal = n_samples - n_faults

    # Normal operation ranges
    voltage_normal = rng.normal(loc=25, scale=0.8, size=n_normal)      # kV
    current_normal = rng.normal(loc=300, scale=40, size=n_normal)      # A
    temp_normal = rng.normal(loc=60, scale=5, size=n_normal)           # °C
    vibration_normal = rng.normal(loc=0.8, scale=0.2, size=n_normal)   # g
    pf_normal = rng.normal(loc=0.95, scale=0.02, size=n_normal)        # power factor
    freq_normal = rng.normal(loc=50, scale=0.1, size=n_normal)         # Hz

    # Faulty operation ranges
    voltage_fault = rng.normal(loc=23, scale=2.0, size=n_faults)
    current_fault = rng.normal(loc=500, scale=80, size=n_faults)
    temp_fault = rng.normal(loc=85, scale=6, size=n_faults)
    vibration_fault = rng.normal(loc=1.6, scale=0.3, size=n_faults)
    pf_fault = rng.normal(loc=0.8, scale=0.05, size=n_faults)
    freq_fault = rng.normal(loc=49.3, scale=0.4, size=n_faults)

    # Combine normal + fault data
    voltage = np.concatenate([voltage_normal, voltage_fault])
    current = np.concatenate([current_normal, current_fault])
    temp = np.concatenate([temp_normal, temp_fault])
    vibration = np.concatenate([vibration_normal, vibration_fault])
    pf = np.clip(np.concatenate([pf_normal, pf_fault]), 0, 1)
    freq = np.concatenate([freq_normal, freq_fault])

    fault_labels = np.concatenate([
        np.zeros(n_normal, dtype=int),   # 0 = normal
        np.ones(n_faults, dtype=int)     # 1 = fault
    ])

    df = pd.DataFrame({
        "line_voltage_kV": voltage,
        "line_current_A": current,
        "transformer_temp_C": temp,
        "vibration_g": vibration,
        "power_factor": pf,
        "frequency_Hz": freq,
        "fault": fault_labels
    })

    return df


if __name__ == "__main__":
    # Quick check
    data = generate_synthetic_data(n_samples=10)
    print(data.head())
