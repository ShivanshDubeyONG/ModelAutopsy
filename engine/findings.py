from __future__ import annotations


def build_findings(
    metrics: dict,
    error_slices: list[dict],
    importance: dict,
) -> list[dict]:

    problem_type = metrics.get(
        "problem_type",
        "classification",
    )

    findings = []

    if problem_type == "regression":
        findings.extend(
            _regression_findings(
                metrics
            )
        )
    else:
        findings.extend(
            _classification_findings(
                metrics
            )
        )

    findings.extend(
        _cohort_findings(
            error_slices
        )
    )

    findings.extend(
        _feature_dependency_findings(
            importance
        )
    )

    findings.sort(
        key=lambda item:
            item.get(
                "score",
                0,
            ),
        reverse=True,
    )

    return findings[:8]


def _classification_findings(
    metrics,
):

    findings = []

    accuracy = metrics.get(
        "accuracy"
    )

    f1 = metrics.get(
        "f1"
    )

    if (
        accuracy is not None
        and accuracy < 0.80
    ):

        severity = (
            "critical"
            if accuracy < 0.60
            else "high"
            if accuracy < 0.70
            else "medium"
        )

        findings.append(
            {
                "severity": severity,
                "type": "performance",
                "title": (
                    "Model performance is weak"
                    if accuracy < 0.70
                    else (
                        "Model performance "
                        "needs attention"
                    )
                ),
                "description": (
                    f"Overall accuracy is "
                    f"{accuracy:.1%}. "
                    f"Weighted F1 is "
                    f"{f1:.1%}."
                    if f1 is not None
                    else (
                        f"Overall accuracy is "
                        f"{accuracy:.1%}."
                    )
                ),
                "score": max(
                    0.0,
                    1.0 - accuracy,
                ),
            }
        )

    return findings


def _regression_findings(
    metrics,
):

    findings = []

    r2 = metrics.get(
        "r2"
    )

    if (
        r2 is not None
        and r2 < 0.50
    ):

        severity = (
            "critical"
            if r2 < 0
            else "high"
        )

        findings.append(
            {
                "severity": severity,
                "type": "performance",
                "title": (
                    "Regression fit is weak"
                ),
                "description": (
                    f"R² is {r2:.2f}; "
                    "the model explains "
                    "limited variance in "
                    "the target."
                ),
                "score": min(
                    1.0,
                    max(
                        0.0,
                        0.5 - r2,
                    ),
                ),
            }
        )

    return findings


def _cohort_findings(
    error_slices,
):

    findings = []

    for item in error_slices:

        lift = float(
            item.get(
                "lift",
                0,
            )
        )

        severity = (
            "critical"
            if lift >= 3
            else "high"
            if lift >= 2
            else "medium"
        )

        if (
            item.get(
                "problem_type"
            )
            == "regression"
        ):

            description = (
                f'{item["condition"]} '
                f'has {lift:.1f}× higher '
                "MAE than the overall "
                "dataset."
            )

        else:

            description = (
                f'{item["condition"]} '
                f'has {lift:.1f}× higher '
                "error rate than the "
                "overall dataset."
            )

        if (
            item.get("kind")
            == "confidence"
        ):

            title = (
                "Confidence-linked "
                "failure cohort"
            )

        elif (
            item.get("kind")
            == "class_specific"
        ):

            title = (
                "Class-specific failure "
                "concentration"
            )

        else:

            title = (
                "High-error cohort detected"
            )

        findings.append(
            {
                "severity": severity,
                "type": "error_cohort",
                "title": title,
                "description": description,
                "feature": item.get(
                    "feature"
                ),
                "evidence": item,
                "score": min(
                    1.0,
                    lift / 3.0,
                ),
            }
        )

    return findings


def _feature_dependency_findings(
    importance,
):

    features = (
        importance.get(
            "features",
            [],
        )
        if isinstance(
            importance,
            dict,
        )
        else []
    )

    if not features:
        return []

    top = features[:3]

    top_mass = sum(
        abs(
            float(
                item.get(
                    "importance",
                    0,
                )
            )
        )
        for item in top
    )

    total_mass = sum(
        abs(
            float(
                item.get(
                    "importance",
                    0,
                )
            )
        )
        for item in features
    )

    if total_mass <= 0:
        return []

    concentration = (
        top_mass
        / total_mass
    )

    if concentration < 0.65:
        return []

    return [
        {
            "severity": "info",
            "type": "feature_dependency",
            "title": (
                "Decision influence "
                "is concentrated"
            ),
            "description": (
                "The top three features "
                f"account for approximately "
                f"{concentration:.0%} of measured "
                "feature influence."
            ),
            "evidence": top,
            "score": (
                0.25
                + 0.35 * concentration
            ),
        }
    ]


def calculate_health_score(
    metrics: dict,
    findings: list[dict],
) -> int:
    """
    Diagnostic health score.

    Important:
        Missing explainability does NOT
        automatically make a model less healthy.

    Health measures observed model behavior,
    not whether SHAP happens to support it.
    """

    problem_type = metrics.get(
        "problem_type",
        "classification",
    )

    if problem_type == "regression":

        r2 = metrics.get(
            "r2"
        )

        if r2 is None:
            return 0

        base = max(
            0.0,
            min(
                100.0,
                50.0
                + 50.0 * r2,
            ),
        )

    else:

        accuracy = metrics.get(
            "accuracy"
        )

        if accuracy is None:
            return 0

        base = (
            accuracy
            * 100.0
        )

    penalties = {
        "critical": 10,
        "high": 6,
        "medium": 3,
    }

    applied = 0

    for finding in findings:

        penalty = penalties.get(
            finding.get(
                "severity"
            ),
            0,
        )

        applied += penalty

        if applied >= 24:
            break

    return int(
        max(
            0,
            min(
                100,
                round(
                    base
                    - applied
                ),
            ),
        )
    )