import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  Crosshair,
  Database,
  Fingerprint,
  Gauge,
  Layers3,
  ShieldAlert,
  Target,
  TrendingDown,
  TrendingUp,
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

function value(
  item: unknown,
): string {
  if (
    item === null ||
    item === undefined
  ) {
    return "—";
  }

  if (
    typeof item === "number"
  ) {
    return Number.isInteger(item)
      ? String(item)
      : item.toFixed(3);
  }

  return String(item);
}

function percent(
  item: unknown,
): string {
  if (
    typeof item !== "number" ||
    !Number.isFinite(item)
  ) {
    return "—";
  }

  return `${(
    item * 100
  ).toFixed(1)}%`;
}

function healthClass(
  score: number,
): string {
  if (score >= 80) return "healthy";
  if (score >= 60) return "watch";
  if (score >= 40) return "risk";

  return "critical";
}

function healthLabel(
  score: number,
): string {
  if (score >= 80) return "Healthy";
  if (score >= 60) return "Watch";
  if (score >= 40) return "At Risk";

  return "Critical";
}

function buildOutputs(
  report: Report,
): OutputReportData[] {
  if (
    report.outputs &&
    report.outputs.length
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
  const outputs =
    buildOutputs(report);

  if (!outputs.length) {
    return (
      <div className="report-empty">
        <AlertTriangle size={18} />

        <div>
          <strong>
            No analyzable outputs
          </strong>

          <p>
            The forensic engine returned
            no output analysis.
          </p>
        </div>
      </div>
    );
  }

  return (
    <main className="ma-report">
      <ReportHero
        report={report}
        outputs={outputs}
      />

      <div className="ma-output-list">
        {outputs.map(
          (output, index) => (
            <OutputSection
              key={`${output.name}-${index}`}
              output={output}
              index={index}
            />
          ),
        )}
      </div>

      <footer className="ma-footer">
        <div>
          <strong>
            MODEL AUTOPSY
          </strong>

          <span>
            Forensic debugging for machine
            learning systems.
          </span>
        </div>

        <span>
          Analysis complete
        </span>
      </footer>
    </main>
  );
}


/* =========================================================
   HERO
   ========================================================= */

function ReportHero({
  report,
  outputs,
}: {
  report: Report;
  outputs: OutputReportData[];
}) {
  const score =
    report.health_score;

  return (
    <section className="ma-hero">
      <div className="ma-hero-main">
        <div className="ma-eyebrow">
          <Fingerprint size={13} />
          FORENSIC MODEL REPORT
        </div>

        <div className="ma-hero-title-row">
          <div>
            <h1>
              {report.model.name}
            </h1>

            <p>
              Forensic analysis of{" "}
              <strong>
                {outputs.length}
              </strong>{" "}
              model output
              {outputs.length === 1
                ? ""
                : "s"}
              .
            </p>
          </div>

          <div
            className={`ma-health ${healthClass(
              score,
            )}`}
          >
            <div className="ma-health-label">
              MODEL HEALTH
            </div>

            <div className="ma-health-value">
              {Math.round(score)}
              <small>
                /100
              </small>
            </div>

            <div className="ma-health-status">
              {healthLabel(score)}
            </div>

            <div className="ma-health-track">
              <i
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(
                      100,
                      score,
                    ),
                  )}%`,
                }}
              />
            </div>
          </div>
        </div>

        <div className="ma-meta">
          <Meta
            icon={<Database size={13} />}
            label="DATASET"
            value={`${report.dataset.samples} samples`}
          />

          <Meta
            icon={<Layers3 size={13} />}
            label="FEATURES"
            value={`${report.dataset.features} features`}
          />

          <Meta
            icon={<Target size={13} />}
            label="OUTPUTS"
            value={`${outputs.length} output${
              outputs.length === 1
                ? ""
                : "s"
            }`}
          />

          <Meta
            icon={<Gauge size={13} />}
            label="ENGINE"
            value="Analysis complete"
          />
        </div>
      </div>
    </section>
  );
}

function Meta({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="ma-meta-item">
      <div className="ma-meta-icon">
        {icon}
      </div>

      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

/* =========================================================
   OUTPUT
   ========================================================= */

function OutputSection({
  output,
  index,
}: {
  output: OutputReportData;
  index: number;
}) {
  return (
    <section
      className="ma-output"
      id={`output-${index}`}
    >
      <OutputHeading
        output={output}
        index={index}
      />

      <Metrics
        output={output}
      />

      <div className="ma-analysis-grid">
        <Findings
          findings={output.findings}
        />

        <FeatureImportance
          output={output}
        />
      </div>

      <FailureCohorts
        output={output}
      />

      <RepresentativeCase
        output={output}
      />

      <Counterfactual
        output={output}
      />
    </section>
  );
}

function OutputHeading({
  output,
  index,
}: {
  output: OutputReportData;
  index: number;
}) {
  return (
    <header className="ma-output-heading">
      <div>
        <div className="ma-section-kicker">
          OUTPUT 0{index + 1}
        </div>

        <h2>
          {prettyName(
            output.name,
          )}
        </h2>

        <p>
          {output.problem_type ===
          "classification"
            ? "Classification decision analysis."
            : "Continuous prediction error analysis."}
        </p>
      </div>

      <div className="ma-output-health">
        <span>
          OUTPUT HEALTH
        </span>

        <strong>
          {Math.round(
            output.health_score,
          )}
        </strong>

        <em>
          {healthLabel(
            output.health_score,
          )}
        </em>
      </div>
    </header>
  );
}


/* =========================================================
   METRICS
   ========================================================= */

function Metrics({
  output,
}: {
  output: OutputReportData;
}) {
  const metrics =
    output.metrics;

  const cards =
    output.problem_type ===
    "classification"
      ? [
          [
            "Accuracy",
            percent(
              metrics.accuracy,
            ),
          ],
          [
            "Precision",
            percent(
              metrics.precision,
            ),
          ],
          [
            "Recall",
            percent(
              metrics.recall,
            ),
          ],
          [
            "F1",
            percent(metrics.f1),
          ],
          [
            "ROC-AUC",
            percent(
              metrics.roc_auc,
            ),
          ],
        ]
      : [
          [
            "MAE",
            value(metrics.mae),
          ],
          [
            "RMSE",
            value(metrics.rmse),
          ],
          [
            "R²",
            value(metrics.r2),
          ],
          [
            "MAPE",
            percent(
              metrics.mape,
            ),
          ],
        ];

  return (
    <div className="ma-metrics">
      {cards.map(
        ([label, metric]) => (
          <div
            className="ma-metric"
            key={label}
          >
            <span>
              {label}
            </span>

            <strong>
              {metric}
            </strong>
          </div>
        ),
      )}
    </div>
  );
}


/* =========================================================
   FINDINGS
   ========================================================= */

function Findings({
  findings,
}: {
  findings: Finding[];
}) {
  return (
    <article className="ma-card">
      <CardHeading
        kicker="01 / DIAGNOSIS"
        title="What went wrong"
      />

      {findings.length === 0 ? (
        <EmptyInline>
          No material findings were detected.
        </EmptyInline>
      ) : (
        <div className="ma-findings">
          {findings.map(
            (finding, index) => (
              <div
                className={`ma-finding ${finding.severity}`}
                key={`${finding.title}-${index}`}
              >
                <span className="ma-finding-index">
                  0{index + 1}
                </span>

                <div>
                  <div className="ma-finding-type">
                    {finding.severity}
                    {" · "}
                    {prettyName(
                      finding.type,
                    )}
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
    </article>
  );
}


/* =========================================================
   FEATURE IMPORTANCE
   ========================================================= */

function FeatureImportance({
  output,
}: {
  output: OutputReportData;
}) {
  const features =
    output.feature_importance
      ?.features ?? [];

  const max = Math.max(
    ...features.map(
      (item) =>
        Math.abs(
          item.importance,
        ),
    ),
    1,
  );

  return (
    <article className="ma-card">
      <CardHeading
        kicker="02 / EVIDENCE"
        title="What drives the model"
        right={
          output.feature_importance
            ?.method
        }
      />

      {!features.length ? (
        <EmptyInline>
          Feature importance is unavailable
          for this model.
        </EmptyInline>
      ) : (
        <div className="ma-feature-list">
          {features
            .slice(0, 7)
            .map(
              (
                feature,
                index,
              ) => (
                <div
                  className="ma-feature"
                  key={
                    feature.feature
                  }
                >
                  <div className="ma-feature-top">
                    <span>
                      0{index + 1}
                    </span>

                    <strong>
                      {prettyName(
                        feature.feature,
                      )}
                    </strong>

                    <b>
                      {feature.importance.toFixed(
                        4,
                      )}
                    </b>
                  </div>

                  <div className="ma-feature-track">
                    <i
                      style={{
                        width: `${
                          (Math.abs(
                            feature.importance,
                          ) /
                            max) *
                          100
                        }%`,
                      }}
                    />
                  </div>
                </div>
              ),
            )}
        </div>
      )}
    </article>
  );
}


/* =========================================================
   FAILURE COHORTS
   ========================================================= */

function FailureCohorts({
  output,
}: {
  output: OutputReportData;
}) {
  const slices =
    output.error_analysis ?? [];

  return (
    <article className="ma-card ma-wide">
      <CardHeading
        kicker="03 / ERROR SURFACE"
        title="Where the model fails"
        right={`${slices.length} cohorts`}
      />

      {!slices.length ? (
        <div className="ma-no-cohorts">
          <CheckCircle2 size={17} />

          <div>
            <strong>
              No concentrated failure cohort
              detected.
            </strong>

            <span>
              No subgroup crossed the current
              diagnostic threshold.
            </span>
          </div>
        </div>
      ) : (
        <div className="ma-cohorts">
          {slices
            .slice(0, 6)
            .map(
              (
                slice,
                index,
              ) => (
                <Cohort
                  slice={slice}
                  output={output}
                  index={index}
                  key={`${slice.feature}-${slice.value}-${index}`}
                />
              ),
            )}
        </div>
      )}
    </article>
  );
}

function Cohort({
  slice,
  output,
  index,
}: {
  slice: ErrorSlice;
  output: OutputReportData;
  index: number;
}) {
  const classification =
    output.problem_type ===
    "classification";

  const error =
    classification
      ? slice.error_rate
      : slice.error;

  return (
    <div className="ma-cohort">
      <div className="ma-cohort-top">
        <span>
          COHORT 0{index + 1}
        </span>

        <b>
          {slice.lift
            ? `${slice.lift.toFixed(2)}×`
            : "—"}
        </b>
      </div>

      <strong>
        {prettyName(
          String(
            slice.feature ??
              "Unknown",
          ),
        )}
      </strong>

      <div className="ma-cohort-condition">
        {String(
          slice.value ??
            "Unknown",
        )}
      </div>

      <div className="ma-cohort-bottom">
        <span>
          ERROR
          <b>
            {classification
              ? percent(error)
              : value(error)}
          </b>
        </span>

        <span>
          BASELINE
          <b>
            {classification
              ? percent(
                  slice.baseline_error,
                )
              : value(
                  slice.baseline_error,
                )}
          </b>
        </span>

        <span>
          N
          <b>
            {slice.size ??
              "—"}
          </b>
        </span>
      </div>
    </div>
  );
}


/* =========================================================
   REPRESENTATIVE FAILURE
   ========================================================= */

function RepresentativeCase({
  output,
}: {
  output: OutputReportData;
}) {
  const item =
    output.representative_case;

  const contributions =
    item.contributions ??
    [];

  const classification =
    output.problem_type ===
    "classification";

  return (
    <article className="ma-case">
      <div className="ma-case-heading">
        <div>
          <div className="ma-section-kicker">
            04 / REPRESENTATIVE FAILURE
          </div>

          <h3>
            The case that explains the problem
          </h3>
        </div>

        <span>
          CASE #
          {String(
            item.index + 1,
          ).padStart(3, "0")}
        </span>
      </div>

      <div className="ma-case-summary">
        <div className="ma-verdict actual">
          <span>
            ACTUAL
          </span>

          <strong>
            {value(item.actual)}
          </strong>
        </div>

        <ArrowRight
          className="ma-case-arrow"
          size={22}
        />

        <div className="ma-verdict predicted">
          <span>
            PREDICTED
          </span>

          <strong>
            {value(
              item.prediction,
            )}
          </strong>
        </div>

        {classification &&
        item.probability !==
          null &&
        item.probability !==
          undefined ? (
          <div className="ma-confidence">
            <span>
              CONFIDENCE
            </span>

            <strong>
              {percent(
                item.probability,
              )}
            </strong>

            <small>
              for{" "}
              {value(
                item.probability_label,
              )}
            </small>
          </div>
        ) : null}
      </div>

      <div className="ma-contribution-area">
        <div className="ma-contribution-heading">
          <div>
            <span>
              FEATURE CONTRIBUTIONS
            </span>

            <p>
              Positive values push the prediction
              upward; negative values push it downward.
            </p>
          </div>

          <span>
            {item.method ??
              "Unavailable"}
          </span>
        </div>

        {!contributions.length ? (
          <EmptyInline>
            Local feature contributions are
            unavailable.
          </EmptyInline>
        ) : (
          <div className="ma-contributions">
            {contributions
              .slice(0, 7)
              .map(
                (
                  contribution,
                ) => (
                  <div
                    className="ma-contribution"
                    key={
                      contribution.feature
                    }
                  >
                    <div className="ma-contribution-info">
                      <strong>
                        {prettyName(
                          contribution.feature,
                        )}
                      </strong>

                      <span>
                        value:{" "}
                        {value(
                          contribution.value,
                        )}
                      </span>
                    </div>

                    <div className="ma-contribution-bar">
                      <i
                        className={
                          contribution.contribution >=
                          0
                            ? "positive"
                            : "negative"
                        }
                        style={{
                          width: `${Math.min(
                            100,
                            Math.abs(
                              contribution.contribution,
                            ) * 500,
                          )}%`,
                        }}
                      />
                    </div>

                    <strong
                      className={
                        contribution.contribution >=
                        0
                          ? "positive-text"
                          : "negative-text"
                      }
                    >
                      {contribution.contribution >=
                      0
                        ? "+"
                        : ""}
                      {contribution.contribution.toFixed(
                        4,
                      )}
                    </strong>
                  </div>
                ),
              )}
          </div>
        )}
      </div>
    </article>
  );
}


/* =========================================================
   COUNTERFACTUAL
   ========================================================= */

function Counterfactual({
  output,
}: {
  output: OutputReportData;
}) {
  const cf =
    output.counterfactual;

  return (
    <article className="ma-card ma-wide">
      <CardHeading
        kicker="05 / WHAT-IF"
        title="What would have changed the decision?"
        right={
          cf.found
            ? "FOUND"
            : "NOT AVAILABLE"
        }
      />

      {!cf.found ? (
        <div className="ma-cf-unavailable">
          <ShieldAlert size={17} />

          <div>
            <strong>
              No valid counterfactual found.
            </strong>

            <span>
              {cf.reason ??
                "The engine could not find a valid change that flips this output."}
            </span>
          </div>
        </div>
      ) : (
        <>
          <div className="ma-cf-verdict">
            <span>
              {value(
                cf.original_prediction,
              )}
            </span>

            <ArrowRight size={17} />

            <strong>
              {value(
                cf.final_prediction ??
                  cf.desired_prediction,
              )}
            </strong>
          </div>

          <div className="ma-cf-changes">
            {(
              cf.changes ?? []
            ).map(
              (change) => (
                <div
                  className="ma-cf-change"
                  key={
                    change.feature
                  }
                >
                  <strong>
                    {prettyName(
                      change.feature,
                    )}
                  </strong>

                  <span>
                    {value(
                      change.from,
                    )}
                  </span>

                  <ArrowRight
                    size={14}
                  />

                  <b>
                    {value(
                      change.to,
                    )}
                  </b>

                  {change.desired_probability !==
                  undefined ? (
                    <small>
                      {percent(
                        change.desired_probability,
                      )}
                    </small>
                  ) : null}
                </div>
              ),
            )}
          </div>
        </>
      )}
    </article>
  );
}


/* =========================================================
   SHARED
   ========================================================= */

function CardHeading({
  kicker,
  title,
  right,
}: {
  kicker: string;
  title: string;
  right?: string;
}) {
  return (
    <header className="ma-card-heading">
      <div>
        <span>
          {kicker}
        </span>

        <h3>
          {title}
        </h3>
      </div>

      {right ? (
        <b>{right}</b>
      ) : null}
    </header>
  );
}

function EmptyInline({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="ma-empty-inline">
      {children}
    </div>
  );
}