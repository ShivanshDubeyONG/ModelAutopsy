from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.loader import load_model
from engine.counterfactuals import find_counterfactual
from engine.diagnostics import discover_error_slices
from engine.explanations import (
    global_explanation,
    local_explanation,
)
from engine.findings import (
    build_findings,
    calculate_health_score,
)
from engine.metrics import (
    calculate_metrics,
    infer_problem_type,
)


def run_autopsy(
    model_path: str | Path,
    dataset_path: str | Path,
    target_column: str | list[str],
) -> dict:
    """
    Run Model Autopsy on a single-output or multi-output model.

    Single output:
        target_column="target"

    Multi-output:
        target_column=["target_a", "target_b"]
    """

    model = load_model(model_path)
    data = pd.read_csv(dataset_path)

    target_columns = _normalise_target_columns(
        target_column
    )

    missing_targets = [
        column
        for column in target_columns
        if column not in data.columns
    ]

    if missing_targets:
        raise ValueError(
            f"Target column(s) not found in dataset: "
            f"{missing_targets}"
        )

    X = data.drop(
        columns=target_columns
    )

    y_frame = data[target_columns]

    predictions = np.asarray(
        model.predict(X)
    )

    prediction_matrix = _normalise_predictions(
        predictions,
        len(data),
    )

    if (
        prediction_matrix.shape[1]
        != len(target_columns)
    ):
        raise ValueError(
            "Number of model outputs does not match "
            f"number of target columns. "
            f"Targets={len(target_columns)}, "
            f"Predictions={prediction_matrix.shape[1]}"
        )

    n_outputs = len(target_columns)

    # =========================================================
    # SINGLE OUTPUT
    # =========================================================

    if n_outputs == 1:
        return _run_single_output(
            model=model,
            X=X,
            y=y_frame.iloc[:, 0].to_numpy(),
            predictions=prediction_matrix[:, 0],
            target_column=target_columns[0],
            data=data,
        )

    # =========================================================
    # MULTI OUTPUT
    # =========================================================

    probabilities = _extract_probabilities(
        model,
        X,
        n_outputs,
    )

    outputs = []

    for output_index, target_name in enumerate(
        target_columns
    ):
        y = y_frame.iloc[
            :,
            output_index,
        ].to_numpy()

        y_pred = prediction_matrix[
            :,
            output_index,
        ]

        # Mixed-output bundles explicitly define
        # the problem type for every output.
        if hasattr(model, "problem_types"):
            if target_name not in model.problem_types:
                raise ValueError(
                    f"No problem type configured for "
                    f"output '{target_name}'."
                )

            problem_type = model.problem_types[
                target_name
            ]

        else:
            problem_type = infer_problem_type(
                y,
                y_pred,
            )

        output_probabilities = probabilities[
            output_index
        ]

        output_report = _analyse_output(
            model=model,
            X=X,
            y=y,
            predictions=y_pred,
            probabilities=output_probabilities,
            problem_type=problem_type,
            output_index=output_index,
            target_name=target_name,
        )

        # IMPORTANT:
        # This must stay INSIDE the loop.
        outputs.append(
            output_report
        )

    # =========================================================
    # OVERALL HEALTH
    # =========================================================

    health_scores = [
        output["health_score"]
        for output in outputs
    ]

    if health_scores:
        overall_health = round(
            float(
                np.mean(
                    health_scores
                )
            )
        )
    else:
        overall_health = 0

    return {
        "model": {
            "name": type(model).__name__,
            "problem_type": "multi_output",
            "n_outputs": n_outputs,
        },
        "dataset": {
            "samples": len(data),
            "features": len(X.columns),
            "feature_names": list(X.columns),
            "targets": target_columns,
        },
        "health_score": overall_health,
        "outputs": outputs,
    }


def _run_single_output(
    model,
    X,
    y,
    predictions,
    target_column,
    data,
) -> dict:
    """
    Preserve the original single-output report contract.
    """

    problem_type = infer_problem_type(
        y,
        predictions,
    )

    probabilities = None

    if (
        problem_type == "classification"
        and hasattr(
            model,
            "predict_proba",
        )
    ):
        try:
            probabilities = model.predict_proba(
                X
            )
        except Exception:
            probabilities = None

    report = _analyse_output(
        model=model,
        X=X,
        y=y,
        predictions=predictions,
        probabilities=probabilities,
        problem_type=problem_type,
        output_index=None,
        target_name=target_column,
    )

    return {
        "model": {
            "name": type(model).__name__,
            "problem_type": problem_type,
        },
        "dataset": {
            "samples": len(data),
            "features": len(X.columns),
            "feature_names": list(X.columns),
            "target": target_column,
        },
        "health_score": report[
            "health_score"
        ],
        "metrics": report[
            "metrics"
        ],
        "error_analysis": report[
            "error_analysis"
        ],
        "feature_importance": report[
            "feature_importance"
        ],
        "representative_case": report[
            "representative_case"
        ],
        "counterfactual": report[
            "counterfactual"
        ],
        "findings": report[
            "findings"
        ],
    }


def _analyse_output(
    model,
    X,
    y,
    predictions,
    probabilities,
    problem_type,
    output_index,
    target_name,
) -> dict:
    """
    Analyse one output independently.

    Each output receives its own:

        metrics
        error cohorts
        feature evidence
        representative case
        counterfactual
        findings
        health score
    """

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
        problem_type=problem_type,
    )

    error_slices = discover_error_slices(
        X,
        y,
        predictions,
        problem_type=problem_type,
    )

    importance = global_explanation(
        model,
        X,
        output_index=output_index,
    )

    local_index = _representative_prediction_index(
        y,
        predictions,
        problem_type,
    )

    local = local_explanation(
        model,
        X,
        local_index,
        output_index=output_index,
    )

    # =========================================================
    # COUNTERFACTUAL
    # =========================================================

    if (
        problem_type == "classification"
        and output_index is None
    ):
        counterfactual = find_counterfactual(
            model,
            X,
            local_index,
        )

    elif problem_type == "classification":
        counterfactual = {
            "found": False,
            "reason": (
                "Output-specific counterfactual "
                "search for multi-output models "
                "is not implemented yet."
            ),
        }

    else:
        counterfactual = {
            "found": False,
            "reason": (
                "Regression counterfactual search "
                "is not implemented yet."
            ),
        }

    # =========================================================
    # FINDINGS
    # =========================================================

    findings = build_findings(
        metrics,
        error_slices,
        importance,
    )

    health = calculate_health_score(
        metrics,
        findings,
    )

    return {
        "name": target_name,
        "problem_type": problem_type,
        "health_score": health,
        "metrics": metrics,
        "error_analysis": error_slices,
        "feature_importance": importance,
        "representative_case": local,
        "counterfactual": counterfactual,
        "findings": findings,
    }


def _normalise_target_columns(
    target_column: str | list[str],
) -> list[str]:
    if isinstance(
        target_column,
        str,
    ):
        return [target_column]

    if isinstance(
        target_column,
        (list, tuple),
    ):
        targets = list(
            target_column
        )

        if not targets:
            raise ValueError(
                "At least one target column "
                "is required."
            )

        if not all(
            isinstance(
                target,
                str,
            )
            for target in targets
        ):
            raise ValueError(
                "All target columns must "
                "be strings."
            )

        if len(set(targets)) != len(targets):
            raise ValueError(
                "Target columns must be unique."
            )

        return targets

    raise TypeError(
        "target_column must be a string "
        "or a list of strings."
    )


def _normalise_predictions(
    predictions,
    n_samples: int,
) -> np.ndarray:
    array = np.asarray(
        predictions
    )

    if array.ndim == 1:
        array = array.reshape(
            -1,
            1,
        )

    elif array.ndim != 2:
        raise ValueError(
            "Model predictions must be "
            "a 1D or 2D array."
        )

    if array.shape[0] != n_samples:
        raise ValueError(
            "Number of predictions does not "
            "match number of dataset rows."
        )

    return array


def _extract_probabilities(
    model,
    X,
    n_outputs: int,
) -> list:
    """
    Normalise classification probabilities.

    MultiOutputClassifier returns:

        [
            output_1_probabilities,
            output_2_probabilities
        ]

    Single-output classifiers return:

        2D numpy array
    """

    if not hasattr(
        model,
        "predict_proba",
    ):
        return [None] * n_outputs

    try:
        raw = model.predict_proba(
            X
        )
    except Exception:
        return [None] * n_outputs

    if isinstance(
        raw,
        list,
    ):
        if len(raw) != n_outputs:
            return [None] * n_outputs

        return raw

    array = np.asarray(
        raw
    )

    if n_outputs == 1:
        return [array]

    # samples x classes x outputs
    if array.ndim == 3:
        if array.shape[2] == n_outputs:
            return [
                array[
                    :,
                    :,
                    index,
                ]
                for index in range(
                    n_outputs
                )
            ]

        # samples x outputs x classes
        if array.shape[1] == n_outputs:
            return [
                array[
                    :,
                    index,
                    :,
                ]
                for index in range(
                    n_outputs
                )
            ]

    return [None] * n_outputs


def _representative_prediction_index(
    y_true,
    y_pred,
    problem_type: str,
) -> int:
    """
    Select a representative problematic prediction.

    Classification:
        First misclassified example.

    Regression:
        Largest absolute prediction error.
    """

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    if len(y_true) == 0:
        return 0

    if problem_type == "classification":
        for index, (
            true,
            pred,
        ) in enumerate(
            zip(
                y_true,
                y_pred,
            )
        ):
            if true != pred:
                return index

        return 0

    try:
        errors = np.abs(
            y_true.astype(float)
            - y_pred.astype(float)
        )

        return int(
            np.argmax(
                errors
            )
        )

    except (
        ValueError,
        TypeError,
    ):
        return 0