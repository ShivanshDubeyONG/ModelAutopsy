def build_findings(
    metrics: dict,
    error_slices: list[dict],
    importance: dict,
) -> list[dict]:
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

    for item in error_slices:
        lift = item["lift"]

        severity = (
            "critical"
            if lift >= 3
            else "high"
            if lift >= 2
            else "medium"
        )

        findings.append(
            {
                "severity": severity,
                "type": "error_cohort",
                "title": "High-error cohort detected",
                "description": (
                    f'{item["condition"]} has a '
                    f'{lift:.1f}× higher error rate '
                    "than the overall dataset."
                ),
                "feature": item["feature"],
                "evidence": item,
                "score": min(
                    lift / 3,
                    1,
                ),
            }
        )

    top_features = (
        importance.get(
            "features",
            [],
        )[:3]
    )

    if top_features:
        findings.append(
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
        )

    findings.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return findings


def calculate_health_score(
    metrics: dict,
    findings: list[dict],
) -> int:
    accuracy = metrics.get(
        "accuracy",
        0,
    )

    base = accuracy * 100

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