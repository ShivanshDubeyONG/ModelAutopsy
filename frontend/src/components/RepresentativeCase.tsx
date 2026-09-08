import {
  ArrowDownRight,
  ArrowUpRight,
  Fingerprint,
} from "lucide-react";

import type { Report } from "./types";
import {
  formatNumber,
  formatPercent,
  prettyName,
} from "./utils";

type Props = {
  caseData: Report["representative_case"];
};

export default function RepresentativeCase({
  caseData,
}: Props) {
  const positive =
    caseData.contributions.filter(
      (item) =>
        item.contribution > 0,
    );

  const negative =
    caseData.contributions.filter(
      (item) =>
        item.contribution < 0,
    );

  return (
    <section className="report-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <span className="section-number">
            04
          </span>

          <div>
            <div className="section-eyebrow">
              <Fingerprint size={13} />
              REPRESENTATIVE CASE
            </div>

            <h2>
              Inside one prediction.
            </h2>

            <p>
              A concrete decision decomposed
              into the feature contributions
              behind it.
            </p>
          </div>
        </div>
      </div>

      <div className="case-layout">
        <div className="case-card">
          <div className="case-top">
            <span className="micro-label">
              CASE #
              {String(
                caseData.index,
              ).padStart(3, "0")}
            </span>

            <span className="method-pill">
              {caseData.method}
            </span>
          </div>

          <div className="prediction-grid">
            <div>
              <span>PREDICTION</span>

              <strong>
                {caseData.prediction}
              </strong>
            </div>

            <div>
              <span>PROBABILITY</span>

              <strong>
                {caseData.probability !==
                null
                  ? formatPercent(
                      caseData.probability,
                    )
                  : "—"}
              </strong>
            </div>
          </div>

          <p className="case-description">
            This case is used as a concrete
            example of how the model arrived at
            its decision.
          </p>
        </div>

        <div className="case-card contribution-card">
          <div className="contribution-columns">
            <ContributionColumn
              title="PUSHED TOWARD"
              icon={
                <ArrowUpRight
                  size={14}
                />
              }
              items={positive}
              positive
            />

            <ContributionColumn
              title="PUSHED AGAINST"
              icon={
                <ArrowDownRight
                  size={14}
                />
              }
              items={negative}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function ContributionColumn({
  title,
  icon,
  items,
  positive = false,
}: {
  title: string;
  icon: React.ReactNode;
  items: Report["representative_case"]["contributions"];
  positive?: boolean;
}) {
  return (
    <div>
      <div
        className={`contribution-heading ${
          positive
            ? "positive"
            : "negative"
        }`}
      >
        {icon}
        {title}
      </div>

      {items.length ? (
        items.map((item) => (
          <div
            className="contribution-row"
            key={item.feature}
          >
            <div>
              <strong>
                {prettyName(
                  item.feature,
                )}
              </strong>

              <span>
                value:{" "}
                {String(item.value)}
              </span>
            </div>

            <b>
              {item.contribution > 0
                ? "+"
                : ""}
              {formatNumber(
                item.contribution,
              )}
            </b>
          </div>
        ))
      ) : (
        <span className="muted">
          None detected.
        </span>
      )}
    </div>
  );
}