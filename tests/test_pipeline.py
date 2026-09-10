from pathlib import Path

import joblib
import pandas as pd

from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.multioutput import (
    MultiOutputClassifier,
    MultiOutputRegressor,
)
from sklearn.model_selection import train_test_split

from backend.core.pipeline import run_autopsy


def test_classification_pipeline(tmp_path: Path):
    iris = load_iris(as_frame=True)

    X = iris.data
    y = iris.target

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    model_path = tmp_path / "iris_model.joblib"
    data_path = tmp_path / "iris.csv"

    joblib.dump(model, model_path)

    dataset = X_test.copy()
    dataset["target"] = y_test
    dataset.to_csv(data_path, index=False)

    report = run_autopsy(
        model_path,
        data_path,
        "target",
    )

    assert report["model"]["problem_type"] == (
        "classification"
    )

    assert report["dataset"]["samples"] > 0

    assert report["metrics"]["problem_type"] == (
        "classification"
    )

    assert "accuracy" in report["metrics"]
    assert "feature_importance" in report
    assert "findings" in report


def test_regression_pipeline(tmp_path: Path):
    diabetes = load_diabetes(as_frame=True)

    X = diabetes.data
    y = diabetes.target

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model.fit(X_train, y_train)

    model_path = tmp_path / "regression_model.joblib"
    data_path = tmp_path / "regression.csv"

    joblib.dump(model, model_path)

    dataset = X_test.copy()
    dataset["target"] = y_test
    dataset.to_csv(data_path, index=False)

    report = run_autopsy(
        model_path,
        data_path,
        "target",
    )

    assert report["model"]["problem_type"] == (
        "regression"
    )

    assert report["metrics"]["problem_type"] == (
        "regression"
    )

    assert "mae" in report["metrics"]
    assert "rmse" in report["metrics"]
    assert "r2" in report["metrics"]

    assert report["metrics"]["mae"] >= 0
    assert report["metrics"]["rmse"] >= 0

    assert report["dataset"]["samples"] > 0
    assert "feature_importance" in report
    assert "findings" in report
    assert "counterfactual" in report


def test_multi_output_regression(tmp_path: Path):
    diabetes = load_diabetes(as_frame=True)

    X = diabetes.data
    y = pd.DataFrame(
        {
            "target_a": diabetes.target,
            "target_b": diabetes.target * 0.5 + 10,
        }
    )

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=30,
            random_state=42,
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    model.fit(X_train, y_train)

    model_path = (
        tmp_path / "multi_regression_model.joblib"
    )
    data_path = tmp_path / "multi_regression.csv"

    joblib.dump(model, model_path)

    dataset = X_test.copy()
    dataset["target_a"] = y_test["target_a"].values
    dataset["target_b"] = y_test["target_b"].values

    dataset.to_csv(
        data_path,
        index=False,
    )

    report = run_autopsy(
        model_path,
        data_path,
        ["target_a", "target_b"],
    )

    assert report["model"]["problem_type"] == (
        "multi_output"
    )

    assert report["model"]["n_outputs"] == 2

    assert report["dataset"]["targets"] == [
        "target_a",
        "target_b",
    ]

    assert len(report["outputs"]) == 2

    for output in report["outputs"]:
        assert output["problem_type"] == "regression"
        assert "mae" in output["metrics"]
        assert "rmse" in output["metrics"]
        assert "r2" in output["metrics"]
        assert "feature_importance" in output
        assert "findings" in output


def test_multi_output_classification(tmp_path: Path):
    iris = load_iris(as_frame=True)

    X = iris.data

    y = pd.DataFrame(
        {
            "species": iris.target,
            "is_class_two": (
                iris.target == 2
            ).astype(int),
        }
    )

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=30,
            random_state=42,
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y["species"],
    )

    model.fit(X_train, y_train)

    model_path = (
        tmp_path / "multi_classification_model.joblib"
    )
    data_path = tmp_path / "multi_classification.csv"

    joblib.dump(model, model_path)

    dataset = X_test.copy()
    dataset["species"] = y_test["species"].values
    dataset["is_class_two"] = (
        y_test["is_class_two"].values
    )

    dataset.to_csv(
        data_path,
        index=False,
    )

    report = run_autopsy(
        model_path,
        data_path,
        ["species", "is_class_two"],
    )

    assert report["model"]["problem_type"] == (
        "multi_output"
    )

    assert report["model"]["n_outputs"] == 2

    assert len(report["outputs"]) == 2

    for output in report["outputs"]:
        assert output["problem_type"] == (
            "classification"
        )

        assert "accuracy" in output["metrics"]
        assert "feature_importance" in output
        assert "findings" in output

def test_mixed_output_model(
    tmp_path: Path,
):
    diabetes = load_diabetes(
        as_frame=True
    )

    X = diabetes.data

    reference_parameter = (
        diabetes.target
    )

    validity_label = (
        diabetes.target
        > diabetes.target.median()
    ).astype(int)

    y = pd.DataFrame(
        {
            "Reference_Parameter": (
                reference_parameter
            ),
            "Validity_Label": (
                validity_label
            ),
        }
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
        )
    )

    regression_model = (
        RandomForestRegressor(
            n_estimators=30,
            random_state=42,
        )
    )

    classification_model = (
        RandomForestClassifier(
            n_estimators=30,
            random_state=42,
        )
    )

    regression_model.fit(
        X_train,
        y_train["Reference_Parameter"],
    )

    classification_model.fit(
        X_train,
        y_train["Validity_Label"],
    )

    models = {
        "Reference_Parameter": (
            regression_model
        ),
        "Validity_Label": (
            classification_model
        ),
    }

    problem_types = {
        "Reference_Parameter": "regression",
        "Validity_Label": "classification",
    }

    from backend.core.model_bundle import (
        save_model_bundle,
    )

    model_path = (
        tmp_path / "powernext_bundle.joblib"
    )

    save_model_bundle(
        model_path,
        models,
        problem_types,
    )

    data_path = (
        tmp_path / "powernext_data.csv"
    )

    dataset = X_test.copy()

    dataset[
        "Reference_Parameter"
    ] = y_test[
        "Reference_Parameter"
    ].values

    dataset[
        "Validity_Label"
    ] = y_test[
        "Validity_Label"
    ].values

    dataset.to_csv(
        data_path,
        index=False,
    )

    report = run_autopsy(
        model_path,
        data_path,
        [
            "Reference_Parameter",
            "Validity_Label",
        ],
    )

    assert report["model"][
        "problem_type"
    ] == "multi_output"

    assert report["model"][
        "n_outputs"
    ] == 2

    assert len(report["outputs"]) == 2

    outputs = {
        output["name"]: output
        for output in report["outputs"]
    }

    assert outputs[
        "Reference_Parameter"
    ]["problem_type"] == "regression"

    assert outputs[
        "Validity_Label"
    ]["problem_type"] == "classification"

    assert (
        "mae"
        in outputs[
            "Reference_Parameter"
        ]["metrics"]
    )

    assert (
        "accuracy"
        in outputs[
            "Validity_Label"
        ]["metrics"]
    )

    assert (
        "r2"
        in outputs[
            "Reference_Parameter"
        ]["metrics"]
    )

    assert (
        "findings"
        in outputs[
            "Reference_Parameter"
        ]
    )

    assert (
        "findings"
        in outputs[
            "Validity_Label"
        ]
    )

    assert 0 <= report[
        "health_score"
    ] <= 100