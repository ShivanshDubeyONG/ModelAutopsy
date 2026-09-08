import { Target } from "lucide-react";

import type { Metrics } from "./types";

type Props = {
  metrics: Metrics;
};

export default function ConfusionMatrix({
  metrics,
}: Props) {
  const matrix =
    metrics.confusion_matrix;

  const labels = metrics.labels;

  const tn = matrix[0]?.[0] ?? 0;
  const fp = matrix[0]?.[1] ?? 0;
  const fn = matrix[1]?.[0] ?? 0;
  const tp = matrix[1]?.[1] ?? 0;

  return (
    <section className="report-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <span className="section-number">
            06
          </span>

          <div>
            <div className="section-eyebrow">
              <Target size={13} />
              CONFUSION MATRIX
            </div>

            <h2>
              Prediction surface.
            </h2>

            <p>
              Actual labels versus the model's
              predicted labels.
            </p>
          </div>
        </div>
      </div>

      <div className="matrix-layout">
        <div className="matrix-card">
          <div className="matrix-axis">
            <span>
              ACTUAL ↓
            </span>

            <span>
              PREDICTED →
            </span>
          </div>

          <div className="matrix">
            <div />

            <div className="matrix-label">
              {labels[0]}
            </div>

            <div className="matrix-label">
              {labels[1]}
            </div>

            <div className="matrix-label side">
              {labels[0]}
            </div>

            <MatrixCell
              label="TRUE NEGATIVE"
              value={tn}
            />

            <MatrixCell
              label="FALSE POSITIVE"
              value={fp}
              error
            />

            <div className="matrix-label side">
              {labels[1]}
            </div>

            <MatrixCell
              label="FALSE NEGATIVE"
              value={fn}
              error
            />

            <MatrixCell
              label="TRUE POSITIVE"
              value={tp}
            />
          </div>
        </div>

        <div className="matrix-summary">
          <SummaryItem
            label="TRUE NEGATIVES"
            value={tn}
          />

          <SummaryItem
            label="FALSE POSITIVES"
            value={fp}
          />

          <SummaryItem
            label="FALSE NEGATIVES"
            value={fn}
          />

          <SummaryItem
            label="TRUE POSITIVES"
            value={tp}
          />
        </div>
      </div>
    </section>
  );
}

function MatrixCell({
  label,
  value,
  error = false,
}: {
  label: string;
  value: number;
  error?: boolean;
}) {
  return (
    <div
      className={`matrix-cell ${
        error ? "error" : ""
      }`}
    >
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SummaryItem({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}