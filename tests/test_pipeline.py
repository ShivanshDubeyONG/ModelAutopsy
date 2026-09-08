import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from backend.core.pipeline import run_autopsy


def test_autopsy_pipeline():
    iris = load_iris(
        as_frame=True
    )

    data = iris.frame.copy()
    data["target"] = iris.target

    X = data.drop(
        columns=["target"]
    )

    y = data["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
    )

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)

        model_path = (
            directory / "model.joblib"
        )

        dataset_path = (
            directory / "test.csv"
        )

        joblib.dump(
            model,
            model_path,
        )

        test_data = X_test.copy()
        test_data["target"] = y_test.values

        test_data.to_csv(
            dataset_path,
            index=False,
        )

        report = run_autopsy(
            model_path,
            dataset_path,
            "target",
        )

        assert report["dataset"]["samples"] > 0
        assert (
            "accuracy"
            in report["metrics"]
        )
        assert (
            "feature_importance"
            in report
        )
        assert (
            "findings"
            in report
        )