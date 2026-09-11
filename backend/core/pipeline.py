from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backend.core.loader import (
    load_model,
)

from engine.counterfactuals import (
    find_counterfactual,
)

from engine.diagnostics import (
    discover_error_slices,
)

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

from engine.model_adapter import (
    get_model_adapter,
)


def run_autopsy(
    model_path: str | Path,
    dataset_path: str | Path,
    target_column: str | list[str] | None,
) -> dict:

    model = load_model(
        model_path
    )

    data = _read_dataset(
        dataset_path
    )

    target_columns = (
        _normalise_target_columns(
            target_column
        )
    )

    missing_targets = [
        column
        for column in target_columns
        if column not in data.columns
    ]

    if missing_targets:
        raise ValueError(
            "Target column(s) not found "
            f"in dataset: {missing_targets}"
        )

    if target_columns:

        X = data.drop(
            columns=target_columns
        )

    else:

        X = data.copy()

    predictions = np.asarray(
        model.predict(X)
    )

    prediction_matrix = (
        _normalise_predictions(
            predictions,
            len(data),
        )
    )

    # ---------------------------------------------------------
    # UNLABELED DATA
    # ---------------------------------------------------------

    if not target_columns:
        return _run_unlabeled(
            model,
            X,
            prediction_matrix,
        )

    # ---------------------------------------------------------
    # OUTPUT COUNT VALIDATION
    # ---------------------------------------------------------

    if (
        prediction_matrix.shape[1]
        != len(target_columns)
    ):
        raise ValueError(
            "Number of model outputs does "
            "not match number of target "
            f"columns. Targets="
            f"{len(target_columns)}, "
            f"Predictions="
            f"{prediction_matrix.shape[1]}"
        )

    # ---------------------------------------------------------
    # SINGLE OUTPUT
    # ---------------------------------------------------------

    if len(target_columns) == 1:

        return _run_single_output(
            model,
            X,
            data,
            target_columns[0],
            prediction_matrix[:, 0],
        )

    # ---------------------------------------------------------
    # MULTI OUTPUT
    # ---------------------------------------------------------

    outputs = []

    for (
        output_index,
        target_name,
    ) in enumerate(
        target_columns
    ):

        y = data[
            target_name
        ].to_numpy()

        y_pred = (
            prediction_matrix[
                :,
                output_index,
            ]
        )

        adapter = get_model_adapter(
            model,
            output_index,
        )

        problem_type = (
            _configured_problem_type(
                model,
                target_name,
                y,
                y_pred,
            )
        )

        probabilities = None

        if (
            problem_type
            == "classification"
        ):

            probabilities = (
                adapter.predict_proba(
                    X
                )
            )

        outputs.append(
            _analyse_output(
                model=model,
                X=X,
                y=y,
                predictions=y_pred,
                probabilities=probabilities,
                problem_type=problem_type,
                output_index=output_index,
                target_name=target_name,
            )
        )

    health_scores = [
        output[
            "health_score"
        ]
        for output in outputs
    ]

    return {
        "model": {
            "name": type(
                model
            ).__name__,
            "problem_type": "multi_output",
            "n_outputs": len(
                target_columns
            ),
        },

        "dataset": {
            "samples": len(data),
            "features": len(
                X.columns
            ),
            "feature_names": list(
                X.columns
            ),
            "targets": target_columns,
        },

        "health_score": (
            round(
                float(
                    np.mean(
                        health_scores
                    )
                )
            )
            if health_scores
            else 0
        ),

        "outputs": outputs,
    }


def _run_single_output(
    model,
    X,
    data,
    target_name,
    predictions,
):

    y = data[
        target_name
    ].to_numpy()

    problem_type = (
        _configured_problem_type(
            model,
            target_name,
            y,
            predictions,
        )
    )

    probabilities = None

    if (
        problem_type
        == "classification"
    ):

        adapter = (
            get_model_adapter(
                model
            )
        )

        probabilities = (
            adapter.predict_proba(
                X
            )
        )

    report = _analyse_output(
        model=model,
        X=X,
        y=y,
        predictions=predictions,
        probabilities=probabilities,
        problem_type=problem_type,
        output_index=None,
        target_name=target_name,
    )

    return {
        "model": {
            "name": type(
                model
            ).__name__,
            "problem_type": problem_type,
        },

        "dataset": {
            "samples": len(data),
            "features": len(
                X.columns
            ),
            "feature_names": list(
                X.columns
            ),
            "target": target_name,
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


def _run_unlabeled(
    model,
    X,
    prediction_matrix,
):

    output_names = list(
        getattr(
            model,
            "output_names",
            [],
        )
    )

    if not output_names:

        output_names = [
            f"output_{index + 1}"
            for index in range(
                prediction_matrix.shape[1]
            )
        ]

    outputs = []

    for index, name in enumerate(
        output_names[
            :prediction_matrix.shape[1]
        ]
    ):

        output_index = (
            index
            if prediction_matrix.shape[1] > 1
            else None
        )

        adapter = (
            get_model_adapter(
                model,
                output_index,
            )
        )

        prediction = (
            prediction_matrix[
                :,
                index,
            ]
        )

        problem_type = (
            _infer_from_model(
                adapter,
                prediction,
            )
        )

        outputs.append(
            {
                "name": name,

                "problem_type": (
                    problem_type
                ),

                "health_score": None,

                "metrics": {
                    "problem_type": (
                        problem_type
                    ),
                    "status": (
                        "unavailable"
                    ),
                    "reason": (
                        "Ground truth was "
                        "not provided."
                    ),
                },

                "error_analysis": [],

                "feature_importance": (
                    global_explanation(
                        model,
                        X,
                        output_index,
                    )
                ),

                "representative_case": (
                    local_explanation(
                        model,
                        X,
                        0,
                        output_index,
                    )
                ),

                "counterfactual": {
                    "found": False,
                    "reason": (
                        "Counterfactuals "
                        "require a labeled "
                        "evaluation target."
                    ),
                },

                "findings": [
                    {
                        "severity": "info",
                        "type": "evaluation",
                        "title": (
                            "Unlabeled "
                            "inference data"
                        ),
                        "description": (
                            "Predictions were "
                            "generated, but "
                            "model error cannot "
                            "be measured without "
                            "ground truth."
                        ),
                        "score": 0.0,
                    }
                ],
            }
        )

    return {
        "model": {
            "name": type(
                model
            ).__name__,
            "problem_type": "inference",
            "n_outputs": len(
                outputs
            ),
        },

        "dataset": {
            "samples": len(X),
            "features": len(
                X.columns
            ),
            "feature_names": list(
                X.columns
            ),
            "targets": [],
        },

        "health_score": None,

        "evaluation": "unlabeled",

        "outputs": outputs,
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
):

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    metrics = calculate_metrics(
        y,
        predictions,
        probabilities,
        problem_type=problem_type,
    )

    # ---------------------------------------------------------
    # ERROR COHORTS
    # ---------------------------------------------------------

    error_slices = (
        discover_error_slices(
            X,
            y,
            predictions,
            problem_type=problem_type,
            probabilities=probabilities,
        )
    )

    # ---------------------------------------------------------
    # GLOBAL EXPLANATION
    # ---------------------------------------------------------

    importance = global_explanation(
        model,
        X,
        output_index=output_index,
        y=y,
        problem_type=problem_type,
    )

    # ---------------------------------------------------------
    # REPRESENTATIVE FAILURE
    # ---------------------------------------------------------

    local_index = (
        _representative_prediction_index(
            y,
            predictions,
            problem_type,
        )
    )

    local = local_explanation(
        model,
        X,
        local_index,
        output_index=output_index,
        actual=y[local_index],
    )

    # ---------------------------------------------------------
    # COUNTERFACTUAL
    # ---------------------------------------------------------

    counterfactual = find_counterfactual(
        model,
        X,
        local_index,
        output_index=output_index,
    )

    # ---------------------------------------------------------
    # FINDINGS
    # ---------------------------------------------------------

    findings = build_findings(
        metrics,
        error_slices,
        importance,
    )

    health = (
        calculate_health_score(
            metrics,
            findings,
        )
    )

    return {
        "name": target_name,

        "problem_type": problem_type,

        "health_score": health,

        "metrics": metrics,

        "error_analysis": (
            error_slices
        ),

        "feature_importance": (
            importance
        ),

        "representative_case": (
            local
        ),

        "counterfactual": (
            counterfactual
        ),

        "findings": findings,
    }


def _configured_problem_type(
    model,
    target_name,
    y,
    predictions,
):

    problem_types = getattr(
        model,
        "problem_types",
        None,
    )

    if (
        isinstance(
            problem_types,
            dict,
        )
        and target_name
        in problem_types
    ):

        return problem_types[
            target_name
        ]

    adapter = get_model_adapter(
        model
    )

    return _infer_from_model(
        adapter,
        predictions,
        y,
    )


def _infer_from_model(
    adapter,
    predictions,
    y=None,
):

    classes = (
        adapter.classes_
    )

    if (
        classes is not None
        and len(classes) >= 2
    ):
        return "classification"

    if y is not None:

        return infer_problem_type(
            y,
            predictions,
        )

    try:

        return infer_problem_type(
            predictions,
            predictions,
        )

    except Exception:

        return "regression"


def _normalise_target_columns(
    target_column,
):

    if (
        target_column is None
        or target_column == ""
    ):
        return []

    if isinstance(
        target_column,
        str,
    ):

        targets = [
            item.strip()
            for item
            in target_column.split(
                ","
            )
            if item.strip()
        ]

    elif isinstance(
        target_column,
        (list, tuple),
    ):

        targets = list(
            target_column
        )

    else:

        raise TypeError(
            "target_column must be "
            "a string, list of strings, "
            "or None."
        )

    if not targets:
        return []

    if not all(
        isinstance(
            item,
            str,
        )
        and item.strip()
        for item in targets
    ):

        raise ValueError(
            "All target columns must "
            "be non-empty strings."
        )

    if len(
        set(targets)
    ) != len(targets):

        raise ValueError(
            "Target columns must "
            "be unique."
        )

    return targets


def _read_dataset(
    path,
):

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: "
            f"{path}"
        )

    try:

        # sep=None + python engine automatically
        # detects comma, semicolon, tab, etc.
        data = pd.read_csv(
            path,
            sep=None,
            engine="python",
        )

    except Exception as exc:

        raise ValueError(
            "Could not parse dataset CSV: "
            f"{exc}"
        ) from exc

    if data.empty:
        raise ValueError(
            "Dataset contains no rows."
        )

    data.columns = [
        str(column).strip()
        for column
        in data.columns
    ]

    return data


def _normalise_predictions(
    predictions,
    n_samples,
):

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

    if (
        array.shape[0]
        != n_samples
    ):

        raise ValueError(
            "Number of predictions "
            "does not match number "
            "of dataset rows."
        )

    return array


def _representative_prediction_index(
    y_true,
    y_pred,
    problem_type,
):

    if len(y_true) == 0:
        return 0

    if problem_type == "classification":

        wrong = np.flatnonzero(
            np.asarray(
                y_true
            )
            != np.asarray(
                y_pred
            )
        )

        if len(wrong):
            return int(
                wrong[0]
            )

        return 0

    try:

        errors = np.abs(
            np.asarray(
                y_true,
                dtype=float,
            )
            - np.asarray(
                y_pred,
                dtype=float,
            )
        )

        return int(
            np.argmax(
                errors
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        return 0