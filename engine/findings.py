def build_findings(
    metrics: dict,
    error_slices: list[dict],
    importance: dict,
) -> list[dict]:
    problem_type = metrics.get(
        "problem_type",
        "classification",
    )

    if problem_type == "regression":
        return _regression_findings(
            metrics,
            error_slices,
            importance,
        )

    return _classification_findings(
        metrics,
        error_slices,
        importance,
    )


def _classification_findings(
    metrics,
    error_slices,
    importance,
):
    findings = []

    accuracy = metrics.get(
        "accuracy",
        0,
    )

    if accuracy < 0.80:
        findings.append(
            {
                "severity": "critical",
                "type": "performance",
                "title": "Model performance is weak",
                "description": (
                    f"Overall accuracy is "
                    f"{accuracy:.1%}."
                ),
                "score": 1.0 - accuracy,
            }
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
        key=lambda item: item["score"],
        reverse=True,
    )

    return findings


def _regression_findings(
    metrics,
    error_slices,
    importance,
):
    findings = []

    r2 = metrics.get(
        "r2"
    )

    if r2 is not None and r2 < 0.50:
        findings.append(
            {
                "severity": "critical",
                "type": "performance",
                "title": "Regression performance is weak",
                "description": (
                    f"R² is {r2:.2f}, "
                    "indicating limited "
                    "predictive fit."
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
        key=lambda item: item["score"],
        reverse=True,
    )

    return findings


def _cohort_findings(
    error_slices,
):
    findings = []

    for item in error_slices:
        lift = item["lift"]

        severity = (
            "critical"
            if lift >= 3
            else "high"
            if lift >= 2
            else "medium"
        )

        if item.get(
            "problem_type"
        ) == "regression":
            description = (
                f'{item["condition"]} has '
                f'{lift:.1f}× higher prediction '
                "error than the overall dataset."
            )
        else:
            description = (
                f'{item["condition"]} has a '
                f'{lift:.1f}× higher error rate '
                "than the overall dataset."
            )

        findings.append(
            {
                "severity": severity,
                "type": "error_cohort",
                "title": "High-error cohort detected",
                "description": description,
                "feature": item["feature"],
                "evidence": item,
                "score": min(
                    lift / 3,
                    1,
                ),
            }
        )

    return findings


def _feature_dependency_findings(
    importance,
):
    top_features = (
        importance.get(
            "features",
            [],
        )[:3]
    )

    if not top_features:
        return []

    return [
        {
            "severity": "info",
            "type": "feature_dependency",
            "title": "Strong feature dependency",
            "description": (
                "The model relies most heavily "
                "on a small group of features."
            ),
            "evidence": top_features,
            "score": 0.4,
        }
    ]


def calculate_health_score(
    metrics: dict,
    findings: list[dict],
) -> int:
    problem_type = metrics.get(
        "problem_type",
        "classification",
    )

    if problem_type == "regression":
        r2 = metrics.get(
            "r2",
            0,
        )

        base = max(
            0,
            min(
                100,
                (r2 + 1) * 50,
            ),
        )
    else:
        base = (
            metrics.get(
                "accuracy",
                0,
            )
            * 100
        )

    penalties = {
        "critical": 15,
        "high": 8,
        "medium": 4,
    }

    for finding in findings:
        base -= penalties.get(
            finding["severity"],
            0,
        )

    return max(
        0,
        min(
            100,
            round(base),
        ),
    )