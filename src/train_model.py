from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from simulate_sensors import generate_synthetic_data


def train_and_save_model(
    n_samples: int = 3000,
    fault_ratio: float = 0.2,
    random_state: int = 42,
    model_dir: str = "../models",
    model_name: str = "fault_detector.pkl",
):
    # 1. Generate dataset
    df = generate_synthetic_data(
        n_samples=n_samples,
        fault_ratio=fault_ratio,
        random_state=random_state
    )

    feature_cols = [
        "line_voltage_kV",
        "line_current_A",
        "transformer_temp_C",
        "vibration_g",
        "power_factor",
        "frequency_Hz",
    ]

    X = df[feature_cols]
    y = df["fault"]

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    # 3. Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 4. Evaluation
    y_pred = model.predict(X_test)
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=3))

    # 5. Save model
    model_dir_path = Path(model_dir)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    model_path = model_dir_path / model_name

    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to: {model_path.resolve()}")


if __name__ == "__main__":
    train_and_save_model()
