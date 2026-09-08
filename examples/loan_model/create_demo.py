from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).parent


def generate_data(n_samples: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.integers(21, 65, n_samples)
    income = rng.normal(65000, 22000, n_samples).clip(18000, 180000)
    credit_score = rng.normal(660, 75, n_samples).clip(300, 850)
    debt_ratio = rng.beta(2.2, 5.0, n_samples)
    employment_years = rng.exponential(6, n_samples).clip(0, 35)

    risk_score = (
        0.004 * (credit_score - 600)
        + 0.000008 * (income - 40000)
        - 2.8 * debt_ratio
        + 0.04 * employment_years
        + 0.012 * (age - 30)
    )

    probability = 1 / (1 + np.exp(-risk_score))

    approved = (
        rng.random(n_samples) < probability
    ).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "credit_score": credit_score.round(0),
            "debt_ratio": debt_ratio.round(3),
            "employment_years": employment_years.round(1),
            "approved": approved,
        }
    )


def main():
    data = generate_data()

    X = data.drop(columns=["approved"])
    y = data["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=7,
        random_state=42,
    )

    model.fit(X_train, y_train)

    test_data = X_test.copy()
    test_data["approved"] = y_test.values

    joblib.dump(
        model,
        ROOT / "model.joblib",
    )

    test_data.to_csv(
        ROOT / "test.csv",
        index=False,
    )

    print("Demo model created.")
    print(f"Model: {ROOT / 'model.joblib'}")
    print(f"Dataset: {ROOT / 'test.csv'}")
    print(f"Training samples: {len(X_train)}")
    print(f"Evaluation samples: {len(X_test)}")


if __name__ == "__main__":
    main()