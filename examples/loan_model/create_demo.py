from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent


def generate_data(n_samples: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.integers(21, 65, n_samples)
    income = rng.normal(70000, 20000, n_samples).clip(18000, 180000)
    credit_score = rng.normal(690, 55, n_samples).clip(300, 850)
    debt_ratio = rng.beta(2.0, 7.0, n_samples)
    employment_years = rng.exponential(6, n_samples).clip(0, 35)

    # Main approval signal.
    score = (
        0.018 * (credit_score - 650)
        + 0.000018 * (income - 50000)
        - 5.0 * debt_ratio
        + 0.08 * employment_years
        + 0.025 * (age - 30)
    )

    # Create a realistic nonlinear interaction.
    score += np.where(
        (credit_score > 720) & (debt_ratio < 0.20),
        1.2,
        0.0,
    )

    probability = 1 / (1 + np.exp(-score))

    approved = (
        rng.random(n_samples) < probability
    ).astype(int)

    # Deliberate hidden cohort:
    # younger applicants with high debt ratios are harder for
    # the model to classify because this group has additional noise.
    difficult_cohort = (
        (age < 30)
        & (debt_ratio > 0.30)
        & (credit_score > 620)
    )

    noisy_probability = probability.copy()

    noisy_probability[difficult_cohort] = (
        0.55 * noisy_probability[difficult_cohort]
        + 0.45 * rng.random(difficult_cohort.sum())
    )

    approved[difficult_cohort] = (
        rng.random(difficult_cohort.sum())
        < noisy_probability[difficult_cohort]
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
        n_estimators=200,
        max_depth=9,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    test_data = X_test.copy()
    test_data["approved"] = y_test.values

    joblib.dump(model, ROOT / "model.joblib")
    test_data.to_csv(ROOT / "test.csv", index=False)

    print("Demo model created.")
    print(f"Model: {ROOT / 'model.joblib'}")
    print(f"Dataset: {ROOT / 'test.csv'}")
    print(f"Training samples: {len(X_train)}")
    print(f"Evaluation samples: {len(X_test)}")


if __name__ == "__main__":
    main()