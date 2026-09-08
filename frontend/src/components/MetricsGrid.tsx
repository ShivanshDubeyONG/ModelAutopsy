import type { Metrics } from "./types";
import {
  formatPercent,
} from "./utils";

type Props = {
  metrics: Metrics;
};

function Metric({
  label,
  value,
}: {
  label: string;
  value: number | undefined;
}) {
  const percentage =
    value === undefined
      ? 0
      : value * 100;

  const state =
    percentage >= 75
      ? "good"
      : percentage >= 60
        ? "watch"
        : "weak";

  return (
    <div className="metric-card">
      <span className="metric-label">
        {label}
      </span>

      <div className="metric-value">
        {formatPercent(value)}
      </div>

      <div className="metric-bar">
        <div className="metric-bar-top">
          <span>
            {label === "ROC-AUC"
              ? "Ranking quality"
              : `${label} score`}
          </span>

          <strong>
            {formatPercent(value)}
          </strong>
        </div>

        <div className="metric-track">
          <div
            className={`metric-fill ${state}`}
            style={{
              width: `${Math.min(
                100,
                Math.max(0, percentage),
              )}%`,
            }}
          />
        </div>
      </div>
    </div>
  );
}

export default function MetricsGrid({
  metrics,
}: Props) {
  return (
    <section className="metrics-grid">
      <Metric
        label="ACCURACY"
        value={metrics.accuracy}
      />

      <Metric
        label="PRECISION"
        value={metrics.precision}
      />

      <Metric
        label="RECALL"
        value={metrics.recall}
      />

      <Metric
        label="F1"
        value={metrics.f1}
      />

      <Metric
        label="ROC-AUC"
        value={metrics.roc_auc}
      />
    </section>
  );
}