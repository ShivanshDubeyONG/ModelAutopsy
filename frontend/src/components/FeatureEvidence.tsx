import {
  BrainCircuit,
  Database,
} from "lucide-react";

import type { Report } from "./types";
import {
  formatNumber,
  prettyName,
} from "./utils";

type Props = {
  report: Report;
};

export default function FeatureEvidence({
  report,
}: Props) {
  const features =
    [...report.feature_importance.features]
      .sort(
        (a, b) =>
          b.importance - a.importance,
      )
      .slice(0, 6);

  const max =
    features.length > 0
      ? Math.max(
          ...features.map(
            (feature) =>
              feature.importance,
          ),
        )
      : 1;

  return (
    <section className="report-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <span className="section-number">
            02
          </span>

          <div>
            <div className="section-eyebrow">
              <BrainCircuit size={13} />
              FEATURE EVIDENCE
            </div>

            <h2>
              What drives the model.
            </h2>

            <p>
              Global feature influence across
              the evaluation set.
            </p>
          </div>
        </div>
      </div>

      <div className="evidence-layout">
        <div className="evidence-card">
          <div className="card-heading">
            <div>
              <span className="micro-label">
                GLOBAL IMPORTANCE
              </span>

              <h3>
                Feature influence
              </h3>
            </div>

            <span className="method-pill">
              {report.feature_importance.method}
            </span>
          </div>

          <div className="importance-list">
            {features.map(
              (feature, index) => {
                const width =
                  (feature.importance /
                    max) *
                  100;

                return (
                  <div
                    className="importance-row"
                    key={feature.feature}
                  >
                    <div className="importance-head">
                      <div>
                        <span className="rank">
                          {String(
                            index + 1,
                          ).padStart(2, "0")}
                        </span>

                        <strong>
                          {prettyName(
                            feature.feature,
                          )}
                        </strong>
                      </div>

                      <span>
                        {formatNumber(
                          feature.importance,
                        )}
                      </span>
                    </div>

                    <div className="importance-track">
                      <div
                        className="importance-fill"
                        style={{
                          width: `${width}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              },
            )}
          </div>
        </div>

        <div className="evidence-card dataset-card">
          <div className="card-heading">
            <div>
              <span className="micro-label">
                DATASET PROFILE
              </span>

              <h3>
                Evaluation surface
              </h3>
            </div>

            <Database size={18} />
          </div>

          <div className="dataset-stats">
            <div>
              <span>SAMPLES</span>
              <strong>
                {report.dataset.samples.toLocaleString()}
              </strong>
            </div>

            <div>
              <span>FEATURES</span>
              <strong>
                {report.dataset.features}
              </strong>
            </div>

            <div>
              <span>TARGET</span>
              <strong>
                {prettyName(
                  report.dataset.target,
                )}
              </strong>
            </div>

            <div>
              <span>MODEL</span>
              <strong>
                {prettyName(
                  report.model.problem_type,
                )}
              </strong>
            </div>
          </div>

          <div className="feature-chips">
            {report.dataset.feature_names.map(
              (feature) => (
                <span
                  key={feature}
                >
                  {prettyName(feature)}
                </span>
              ),
            )}
          </div>
        </div>
      </div>
    </section>
  );
}