import {
  Check,
  ShieldAlert,
} from "lucide-react";

import type { Finding } from "./types";
import {
  formatNumber,
  prettyName,
  severityClass,
} from "./utils";

type Props = {
  findings: Finding[];
};

export default function Findings({
  findings,
}: Props) {
  return (
    <section className="report-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <span className="section-number">
            01
          </span>

          <div>
            <div className="section-eyebrow">
              <ShieldAlert size={13} />
              FORENSIC FINDINGS
            </div>

            <h2>
              What the model is telling us.
            </h2>

            <p>
              The highest-signal observations
              extracted from the evaluation run.
            </p>
          </div>
        </div>

        <span className="section-count">
          {findings.length} findings
        </span>
      </div>

      {findings.length ? (
        <div className="findings-grid">
          {findings.map((finding, index) => {
            const severity =
              severityClass(
                finding.severity,
              );

            return (
              <article
                className={`finding-card ${severity}`}
                key={`${finding.type}-${index}`}
              >
                <div className="finding-line" />

                <div className="finding-body">
                  <div className="finding-top">
                    <span
                      className={`severity-badge ${severity}`}
                    >
                      {finding.severity.toUpperCase()}
                    </span>

                    <span className="finding-type">
                      {prettyName(
                        finding.type,
                      )}
                    </span>
                  </div>

                  <h3>{finding.title}</h3>

                  <p>
                    {finding.description}
                  </p>

                  {finding.evidence?.length ? (
                    <div className="finding-evidence">
                      {finding.evidence
                        .slice(0, 3)
                        .map(
                          (
                            item,
                            evidenceIndex,
                          ) => (
                            <div
                              className="finding-evidence-row"
                              key={
                                evidenceIndex
                              }
                            >
                              <span>
                                {prettyName(
                                  item.feature,
                                )}
                              </span>

                              {item.importance !==
                                undefined && (
                                <strong>
                                  {formatNumber(
                                    item.importance,
                                  )}
                                </strong>
                              )}
                            </div>
                          ),
                        )}
                    </div>
                  ) : null}
                </div>
              </article>
            );
          })}
        </div>
      ) : (
        <div className="empty-state">
          <Check size={17} />
          <div>
            <strong>
              No major findings detected.
            </strong>

            <p>
              The current analysis did not
              surface a high-confidence issue.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}