import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BrainCircuit,
  CheckCircle2,
  Crosshair,
  Database,
  Fingerprint,
  Gauge,
  Layers3,
  Target,
} from "lucide-react";

import type {
  ErrorSlice,
  Finding,
  OutputReport as OutputReportData,
  Report,
} from "./types";

type Props = {
  report: Report;
};

function prettyName(
  value: string,
): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) =>
      char.toUpperCase(),
    );
}

function number(
  value: unknown,
  digits = 3,
): string {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value)
  ) {
    return "—";
  }

  return value.toFixed(digits);
}

function percent(
  value: unknown,
): string {
  if (
    typeof value !== "number" ||
    !Number.isFinite(value)
  ) {
    return "—";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function healthClass(
  score: number,
): string {
  if (score >= 80) return "healthy";
  if (score >= 60) return "watch";
  if (score >= 40) return "risk";

  return "critical";
}

function buildOutputs(
  report: Report,
): OutputReportData[] {
  if (
    report.outputs &&
    report.outputs.length > 0
  ) {
    return report.outputs;
  }

  if (
    report.metrics &&
    report.feature_importance &&
    report.representative_case &&
    report.counterfactual &&
    report.findings
  ) {
    return [
      {
        name:
          report.dataset.target ??
          "target",
        problem_type:
          report.model.problem_type,
        health_score:
          report.health_score,
        metrics:
          report.metrics,
        error_analysis:
          report.error_analysis ?? [],
        feature_importance:
          report.feature_importance,
        representative_case:
          report.representative_case,
        counterfactual:
          report.counterfactual,
        findings:
          report.findings,
      },
    ];
  }

  return [];
}

export default function OutputReport({
  report,
}: Props) {
  const outputs = buildOutputs(report);

  if (!outputs.length) {
    return (
      <div className="empty-state">
        <AlertTriangle size={18} />

        <div>
          <strong>
            No output analysis available.
          </strong>

          <p>
            The backend returned a report
            without analyzable outputs.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="autopsy-report">
      <header className="autopsy-report-hero">
        <div className="autopsy-hero-copy">
          <div className="eyebrow">
            <Fingerprint size={12} />
            FORENSIC MODEL REPORT
          </div>

          <h1>
            {report.model.name}
          </h1>

          <p>
            Failure analysis across{" "}
            <strong>
              {outputs.length}
            </strong>{" "}
            model output
            {outputs.length === 1
              ? ""
              : "s"}
            .
          </p>

          <div className="report-stat-strip">
            <span>
              <Database size={12} />
              {report.dataset.samples} samples
            </span>

            <span>
              <Layers3 size={12} />
              {report.dataset.features} features
            </span>

            <span>
              <Target size={12} />
              {outputs.length} output
              {outputs.length === 1
                ? ""
                : "s"}
            </span>

            <span>
              <BrainCircuit size={12} />
              {prettyName(
                report.model.problem_type,
              )}
            </span>
          </div>
        </div>

        <div
          className={`autopsy-health ${healthClass(
            report.health_score,
          )}`}
        >
          <span>
            SYSTEM HEALTH
          </span>

          <strong>
            {Math.round(
              report.health_score,
            )}
          </strong>

          <small>/ 100</small>

          <div className="health-track">
            <i
              style={{
                width: `${Math.max(
                  0,
                  Math.min(
                    100,
                    report.health_score,
                  ),
                )}%`,
              }}
            />
          </div>
        </div>
      </header>

      <OutputTabs outputs={outputs} />

      <OutputPanels outputs={outputs} />
    </div>
  );
}

function OutputTabs({
  outputs,
}: {
  outputs: OutputReportData[];
}) {
  return (
    <div className="output-overview">
      <div className="output-overview-heading">
        <div>
          <div className="eyebrow">
            <Activity size={12} />
            OUTPUT SURFACE
          </div>

          <h2>
            Model outputs under investigation
          </h2>
        </div>

        <span>
          {outputs.length} registered
        </span>
      </div>

      <div className="output-tab-grid">
        {outputs.map((output, index) => (
          <a
            href={`#output-${index}`}
            className="output-tab"
            key={output.name}
          >
            <div className="output-tab-index">
              0{index + 1}
            </div>

            <div className="output-tab-body">
              <strong>
                {prettyName(output.name)}
              </strong>

              <span>
                {output.problem_type.toUpperCase()}
              </span>
            </div>

            <div
              className={`output-tab-score ${healthClass(
                output.health_score,
              )}`}
            >
              {Math.round(
                output.health_score,
              )}
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function OutputPanels({
  outputs,
}: {
  outputs: OutputReportData[];
}) {
  return (
    <div className="output-panels">
      {outputs.map((output, index) => (
        <section
          className="output-panel"
          id={`output-${index}`}
          key={output.name}
        >
          <OutputHeader
            output={output}
            index={index}
          />

          <MetricGrid
            output={output}
          />

          <div className="analysis-grid">
            <FindingsPanel
              findings={output.findings}
            />

            <FeaturePanel
              output={output}
            />
          </div>

          <ErrorPanel
            slices={output.error_analysis}
            problemType={
              output.problem_type
            }
          />

          <RepresentativePanel
            output={output}
          />

          <BehaviorPanel
            output={output}
          />
        </section>
      ))}
    </div>
  );
}

function OutputHeader({
  output,
  index,
}: {
  output: OutputReportData;
  index: number;
}) {
  return (
    <div className="output-panel-header">
      <div>
        <span className="output-index">
          OUTPUT 0{index + 1}
        </span>

        <h2>
          {prettyName(output.name)}
        </h2>

        <p>
          {output.problem_type ===
          "classification"
            ? "Classification decision surface and failure cohorts."
            : "Continuous prediction error surface and contributing factors."}
        </p>
      </div>

      <div
        className={`output-health ${healthClass(
          output.health_score,
        )}`}
      >
        <Gauge size={15} />

        <div>
          <span>OUTPUT HEALTH</span>
          <strong>
            {Math.round(
              output.health_score,
            )}
          </strong>
        </div>
      </div>
    </div>
  );
}

function MetricGrid({
  output,
}: {
  output: OutputReportData;
}) {
  const metrics = output.metrics;

  const classification =
    output.problem_type ===
    "classification";

  const cards = classification
    ? [
        [
          "ACCURACY",
          percent(metrics.accuracy),
        ],
        [
          "PRECISION",
          percent(metrics.precision),
        ],
        [
          "RECALL",
          percent(metrics.recall),
        ],
        [
          "F1",
          percent(metrics.f1),
        ],
        [
          "ROC-AUC",
          percent(metrics.roc_auc),
        ],
      ]
    : [
        [
          "MAE",
          number(metrics.mae),
        ],
        [
          "RMSE",
          number(metrics.rmse),
        ],
        [
          "R²",
          number(metrics.r2),
        ],
        [
          "MAPE",
          percent(metrics.mape),
        ],
        [
          "HEALTH",
          `${Math.round(
            output.health_score,
          )}/100`,
        ],
      ];

  return (
    <div className="forensic-metrics">
      {cards.map(([label, value]) => (
        <div
          className="forensic-metric"
          key={label}
        >
          <span>{label}</span>

          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function FindingsPanel({
  findings,
}: {
  findings: Finding[];
}) {
  return (
    <div className="forensic-card findings-panel">
      <div className="forensic-card-heading">
        <div>
          <span>
            EVIDENCE CHAIN / 01
          </span>

          <h3>
            What the engine found
          </h3>
        </div>

        <AlertTriangle size={16} />
      </div>

      {findings.length === 0 ? (
        <div className="forensic-empty">
          No material findings detected.
        </div>
      ) : (
        <div className="finding-stack">
          {findings.map(
            (finding, index) => (
              <div
                className={`forensic-finding ${finding.severity}`}
                key={`${finding.title}-${index}`}
              >
                <div className="finding-marker">
                  {String(
                    index + 1,
                  ).padStart(2, "0")}
                </div>

                <div>
                  <div className="finding-meta">
                    <span>
                      {finding.severity.toUpperCase()}
                    </span>

                    <span>
                      {prettyName(
                        finding.type,
                      )}
                    </span>
                  </div>

                  <strong>
                    {finding.title}
                  </strong>

                  <p>
                    {finding.description}
                  </p>
                </div>
              </div>
            ),
          )}
        </div>
      )}
    </div>
  );
}

function FeaturePanel({
  output,
}: {
  output: OutputReportData;
}) {
  const features =
    output.feature_importance
      ?.features ?? [];

  const max = Math.max(
    ...features.map(
      (item) => item.importance,
    ),
    1,
  );

  return (
    <div className="forensic-card feature-panel">
      <div className="forensic-card-heading">
        <div>
          <span>
            EVIDENCE CHAIN / 02
          </span>

          <h3>
            Contributing features
          </h3>
        </div>

        <Crosshair size={16} />
      </div>

      <div className="feature-stack">
        {features
          .slice(0, 8)
          .map((feature, index) => (
            <div
              className="forensic-feature"
              key={feature.feature}
            >
              <div className="feature-line">
                <span>
                  {String(
                    index + 1,
                  ).padStart(2, "0")}
                </span>

                <strong>
                  {prettyName(
                    feature.feature,
                  )}
                </strong>

                <b>
                  {number(
                    feature.importance,
                    4,
                  )}
                </b>
              </div>

              <div className="feature-track">
                <i
                  style={{
                    width: `${
                      (feature.importance /
                        max) *
                      100
                    }%`,
                  }}
                />
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

function ErrorPanel({
  slices,
  problemType,
}: {
  slices: ErrorSlice[];
  problemType: string;
}) {
  return (
    <div className="forensic-card error-panel">
      <div className="forensic-card-heading">
        <div>
          <span>
            EVIDENCE CHAIN / 03
          </span>

          <h3>
            Failure cohorts
          </h3>
        </div>

        <span className="panel-count">
          {slices.length} detected
        </span>
      </div>

      {slices.length === 0 ? (
        <div className="forensic-empty">
          No high-risk cohorts crossed the
          diagnostic threshold.
        </div>
      ) : (
        <div className="cohort-grid">
          {slices
            .slice(0, 8)
            .map((slice, index) => {
              const error =
                problemType ===
                "classification"
                  ? slice.error_rate
                  : slice.error;

              return (
                <div
                  className="cohort-card"
                  key={`${slice.feature}-${slice.value}-${index}`}
                >
                  <div className="cohort-top">
                    <span>
                      COHORT{" "}
                      {String(
                        index + 1,
                      ).padStart(2, "0")}
                    </span>

                    <b>
                      {slice.lift
                        ? `${number(
                            slice.lift,
                            2,
                          )}×`
                        : "—"}
                    </b>
                  </div>

                  <h4>
                    {prettyName(
                      String(
                        slice.feature ??
                          "unknown",
                      ),
                    )}
                  </h4>

                  <strong>
                    {String(
                      slice.value ??
                        "unknown",
                    )}
                  </strong>

                  <div className="cohort-stats">
                    <span>
                      ERROR
                      <b>
                        {problemType ===
                        "classification"
                          ? percent(error)
                          : number(error)}
                      </b>
                    </span>

                    <span>
                      BASELINE
                      <b>
                        {problemType ===
                        "classification"
                          ? percent(
                              slice.baseline_error,
                            )
                          : number(
                              slice.baseline_error,
                            )}
                      </b>
                    </span>

                    <span>
                      SAMPLES
                      <b>
                        {slice.size ??
                          "—"}
                      </b>
                    </span>
                  </div>
                </div>
              );
            })}
        </div>
      )}
    </div>
  );
}

function RepresentativePanel({
  output,
}: {
  output: OutputReportData;
}) {
  const item =
    output.representative_case;

  const evidence = [];

  if (
    Array.isArray(item.features)
  ) {
    evidence.push(
      ...item.features.map(
        (feature: any) => ({
          name:
            feature.feature ??
            feature.name ??
            "feature",
          value:
            feature.value ??
            feature.contribution ??
            "—",
        }),
      ),
    );
  } else if (
    item.features &&
    typeof item.features ===
      "object"
  ) {
    evidence.push(
      ...Object.entries(
        item.features as Record<
          string,
          unknown
        >,
      ).map(([name, value]) => ({
        name,
        value,
      })),
    );
  }

  if (
    evidence.length === 0 &&
    item.contributions
  ) {
    evidence.push(
      ...item.contributions.map(
        (item) => ({
          name: item.feature,
          value: item.value,
        }),
      ),
    );
  }

  return (
    <div className="forensic-card representative-panel">
      <div className="forensic-card-heading">
        <div>
          <span>
            EVIDENCE CHAIN / 04
          </span>

          <h3>
            Representative case
          </h3>
        </div>

        <span className="case-id">
          CASE #
          {String(
            item.index,
          ).padStart(3, "0")}
        </span>
      </div>

      <div className="case-summary">
        <div>
          <span>
            ACTUAL
          </span>

          <strong>
            {item.actual !== undefined
              ? String(item.actual)
              : "—"}
          </strong>
        </div>

        <div>
          <span>
            PREDICTION
          </span>

          <strong>
            {String(
              item.prediction,
            )}
          </strong>
        </div>

        <div>
          <span>
            CONFIDENCE
          </span>

          <strong>
            {item.probability !==
            null &&
            item.probability !==
              undefined
              ? percent(
                  item.probability,
                )
              : "—"}
          </strong>
        </div>
      </div>

      {evidence.length > 0 && (
        <div className="case-evidence">
          {evidence
            .slice(0, 8)
            .map((entry) => (
              <div
                key={entry.name}
              >
                <span>
                  {prettyName(
                    entry.name,
                  )}
                </span>

                <strong>
                  {String(
                    entry.value,
                  )}
                </strong>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

function BehaviorPanel({
  output,
}: {
  output: OutputReportData;
}) {
  const counterfactual =
    output.counterfactual;

  const classification =
    output.problem_type ===
    "classification";

  return (
    <div className="behavior-grid">
      <div className="forensic-card behavior-card">
        <div className="forensic-card-heading">
          <div>
            <span>
              EVIDENCE CHAIN / 05
            </span>

            <h3>
              Model behavior
            </h3>
          </div>

          <BrainCircuit size={16} />
        </div>

        <div className="behavior-copy">
          <p>
            {classification
              ? "The model's classification behavior is evaluated against observed labels, confidence, and concentrated failure cohorts."
              : "The model's regression behavior is evaluated through absolute error, cohort-level error concentration, and feature contribution."}
          </p>

          <div className="behavior-tags">
            <span>
              {output.problem_type.toUpperCase()}
            </span>

            <span>
              {output.feature_importance
                ?.method ??
                "ANALYSIS"}
            </span>
          </div>
        </div>
      </div>

      <div className="forensic-card counterfactual-panel">
        <div className="forensic-card-heading">
          <div>
            <span>
              EVIDENCE CHAIN / 06
            </span>

            <h3>
              Counterfactual
            </h3>
          </div>

          {counterfactual.found ? (
            <CheckCircle2
              size={16}
            />
          ) : (
            <AlertTriangle
              size={16}
            />
          )}
        </div>

        {counterfactual.found ? (
          <>
            <div className="cf-values">
              <div>
                <span>
                  ORIGINAL
                </span>

                <strong>
                  {String(
                    counterfactual.original_prediction,
                  )}
                </strong>
              </div>

              <ArrowDownRight
                size={16}
              />

              <div>
                <span>
                  DESIRED
                </span>

                <strong>
                  {String(
                    counterfactual.desired_prediction,
                  )}
                </strong>
              </div>
            </div>

            {counterfactual.changes &&
              counterfactual.changes
                .length > 0 && (
                <div className="cf-changes">
                  {counterfactual.changes.map(
                    (change) => (
                      <div
                        key={
                          change.feature
                        }
                      >
                        <span>
                          {prettyName(
                            change.feature,
                          )}
                        </span>

                        <b>
                          {String(
                            change.from,
                          )}
                          {" → "}
                          {String(
                            change.to,
                          )}
                        </b>
                      </div>
                    ),
                  )}
                </div>
              )}
          </>
        ) : (
          <div className="forensic-empty">
            {counterfactual.reason ??
              "No counterfactual was available for this output."}
          </div>
        )}
      </div>
    </div>
  );
}