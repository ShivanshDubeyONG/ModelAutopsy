import { AlertTriangle } from "lucide-react";

import type { ErrorSlice } from "./types";
import {
  formatPercent,
  prettyName,
} from "./utils";

type Props = {
  slices: ErrorSlice[];
};

export default function ErrorAnalysis({
  slices,
}: Props) {
  return (
    <section className="report-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <span className="section-number">
            03
          </span>

          <div>
            <div className="section-eyebrow">
              <AlertTriangle size={13} />
              ERROR ANALYSIS
            </div>

            <h2>
              Where the model breaks.
            </h2>

            <p>
              Cohorts with unusually concentrated
              prediction errors.
            </p>
          </div>
        </div>
      </div>

      {slices.length ? (
        <div className="slice-grid">
          {slices
            .slice(0, 8)
            .map((slice, index) => (
              <article
                className="slice-card"
                key={index}
              >
                <div className="slice-top">
                  <span>
                    {String(
                      index + 1,
                    ).padStart(2, "0")}
                  </span>

                  <b>
                    {slice.lift !==
                    undefined
                      ? `${slice.lift.toFixed(
                          1,
                        )}× lift`
                      : "Elevated error"}
                  </b>
                </div>

                <h3>
                  {prettyName(
                    slice.feature ||
                      "Error cohort",
                  )}
                </h3>

                {slice.value !==
                  undefined && (
                  <p>
                    {String(
                      slice.value,
                    )}
                  </p>
                )}

                <div className="slice-stats">
                  <div>
                    <span>
                      ERROR RATE
                    </span>

                    <strong>
                      {formatPercent(
                        slice.error_rate,
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>
                      BASELINE
                    </span>

                    <strong>
                      {formatPercent(
                        slice.baseline_error,
                      )}
                    </strong>
                  </div>

                  {slice.size !==
                    undefined && (
                    <div>
                      <span>
                        ROWS
                      </span>

                      <strong>
                        {slice.size}
                      </strong>
                    </div>
                  )}
                </div>
              </article>
            ))}
        </div>
      ) : (
        <div className="empty-state">
          <AlertTriangle size={17} />

          <div>
            <strong>
              No concentrated error cohorts
              detected.
            </strong>

            <p>
              No subgroup had sufficient
              evidence to qualify as a
              high-error slice.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}