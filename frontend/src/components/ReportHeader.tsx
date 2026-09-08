import {
  Activity,
  LogOut,
} from "lucide-react";

import Brand from "./Brand";
import type { Report } from "./types";
import {
  healthClass,
  healthLabel,
  prettyName,
} from "./utils";

type Props = {
  report: Report;
  onNewAutopsy: () => void;
  onLogout: () => void;
};

export default function ReportHeader({
  report,
  onNewAutopsy,
  onLogout,
}: Props) {
  const score = report.health_score;
  const state = healthClass(score);

  return (
    <header className="report-header">
      <div className="report-topbar">
        <Brand
          onClick={() =>
            window.location.reload()
          }
        />

        <div className="report-actions">
          <span className="analysis-status">
            <i />
            ANALYSIS COMPLETE
          </span>

          <button
            className="new-autopsy-button"
            onClick={onNewAutopsy}
          >
            NEW AUTOPSY
          </button>

          <button
            className="logout-button"
            onClick={onLogout}
            title="Sign out"
          >
            <LogOut size={13} />
          </button>
        </div>
      </div>

      <div className="report-hero">
        <div className="report-identity">
          <div className="eyebrow">
            <Activity size={13} />
            AUTOPSY REPORT
          </div>

          <h1>{report.model.name}</h1>

          <div className="report-meta">
            <span>
              {report.dataset.samples.toLocaleString()}{" "}
              samples
            </span>

            <b />

            <span>
              {report.dataset.features} features
            </span>

            <b />

            <span>
              target{" "}
              <strong>
                {prettyName(
                  report.dataset.target,
                )}
              </strong>
            </span>
          </div>
        </div>

        <div className={`health-block ${state}`}>
          <span className="health-caption">
            MODEL HEALTH
          </span>

          <div className="health-value">
            <strong>{score}</strong>

            <span className="health-denominator">
              / 100
            </span>
          </div>

          <span className="health-label">
            {healthLabel(score)}
          </span>
        </div>
      </div>
    </header>
  );
}