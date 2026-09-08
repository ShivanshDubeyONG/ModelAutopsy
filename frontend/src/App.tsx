import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BrainCircuit,
  FileUp,
  ShieldAlert,
  Target,
  Upload,
} from "lucide-react";

type Report = {
  health_score: number;
  model: {
    name: string;
    problem_type: string;
  };
  dataset: {
    samples: number;
    features: number;
    target: string;
  };
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
    roc_auc?: number;
  };
  findings: Array<{
    severity: string;
    title: string;
    description: string;
    type: string;
  }>;
  feature_importance: {
    features: Array<{
      feature: string;
      importance: number;
    }>;
  };
  representative_case: {
    index: number;
    prediction: string | number;
    probability?: number;
    contributions: Array<{
      feature: string;
      value: string | number;
      contribution: number;
    }>;
  };
  counterfactual: {
    found: boolean;
    original_prediction?: string | number;
    desired_prediction?: string | number;
    changes?: Array<{
      feature: string;
      from: string | number;
      to: string | number;
    }>;
  };
};

function App() {
  const [model, setModel] =
    useState<File | null>(null);

  const [dataset, setDataset] =
    useState<File | null>(null);

  const [target, setTarget] =
    useState("");

  const [report, setReport] =
    useState<Report | null>(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  async function runAutopsy() {
    if (!model || !dataset || !target) {
      setError(
        "Upload a model, dataset, and specify the target column."
      );
      return;
    }

    setLoading(true);
    setError("");

    const form = new FormData();

    form.append("model", model);
    form.append("dataset", dataset);
    form.append("target_column", target);

    try {
      const response = await fetch(
        "http://localhost:8000/api/analyze",
        {
          method: "POST",
          body: form,
        }
      );

      const result = await response.json();

      if (!result.success) {
        throw new Error(
          result.error ||
            "Autopsy failed."
        );
      }

      setReport(result.report);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  }

  if (report) {
    return (
      <Dashboard
        report={report}
        onReset={() =>
          setReport(null)
        }
      />
    );
  }

  return (
    <main className="landing">
      <nav className="nav">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={18} />
          </div>

          <span>
            MODEL
            <strong>AUTOPSY</strong>
          </span>
        </div>

        <span className="status">
          <span className="status-dot" />
          ENGINE ONLINE
        </span>
      </nav>

      <section className="hero">
        <div className="eyebrow">
          MACHINE LEARNING FORENSICS
        </div>

        <h1>
          Find out
          <br />
          <span>why your model fails.</span>
        </h1>

        <p className="hero-copy">
          Upload a trained model and labeled
          evaluation data. Model Autopsy
          investigates the failures, finds
          dangerous cohorts, explains decisions,
          and builds evidence for every finding.
        </p>

        <div className="upload-panel">
          <UploadBox
            label="TRAINED MODEL"
            hint=".joblib / .pkl"
            file={model}
            onFile={setModel}
            accept=".joblib,.pkl"
          />

          <UploadBox
            label="EVALUATION DATA"
            hint=".csv"
            file={dataset}
            onFile={setDataset}
            accept=".csv"
          />

          <div className="target-input">
            <label>TARGET COLUMN</label>

            <input
              value={target}
              onChange={(event) =>
                setTarget(
                  event.target.value
                )
              }
              placeholder="e.g. approved"
            />
          </div>

          <button
            className="autopsy-button"
            onClick={runAutopsy}
            disabled={loading}
          >
            {loading
              ? "RUNNING AUTOPSY..."
              : "BEGIN AUTOPSY"}

            <ArrowRight size={18} />
          </button>

          {error && (
            <div className="error-box">
              <AlertTriangle size={16} />
              {error}
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function UploadBox({
  label,
  hint,
  file,
  onFile,
  accept,
}: {
  label: string;
  hint: string;
  file: File | null;
  onFile: (file: File) => void;
  accept: string;
}) {
  return (
    <label className="upload-box">
      <input
        type="file"
        accept={accept}
        onChange={(event) => {
          const selected =
            event.target.files?.[0];

          if (selected) {
            onFile(selected);
          }
        }}
      />

      <FileUp size={21} />

      <div>
        <strong>{label}</strong>

        <span>
          {file
            ? file.name
            : `Drop or choose ${hint}`}
        </span>
      </div>
    </label>
  );
}

function Dashboard({
  report,
  onReset,
}: {
  report: Report;
  onReset: () => void;
}) {
  return (
    <main className="dashboard">
      <nav className="nav">
        <div className="brand">
          <div className="brand-mark">
            <Activity size={18} />
          </div>

          <span>
            MODEL
            <strong>AUTOPSY</strong>
          </span>
        </div>

        <button
          className="new-autopsy"
          onClick={onReset}
        >
          NEW AUTOPSY
        </button>
      </nav>

      <section className="dashboard-header">
        <div>
          <div className="eyebrow">
            AUTOPSY REPORT
          </div>

          <h1>
            {report.model.name}
          </h1>

          <p>
            {report.dataset.samples.toLocaleString()}
            {" "}samples ·{" "}
            {report.dataset.features}
            {" "}features · target{" "}
            <code>
              {report.dataset.target}
            </code>
          </p>
        </div>

        <div className="health">
          <span>MODEL HEALTH</span>

          <strong>
            {report.health_score}
          </strong>

          <small>/ 100</small>
        </div>
      </section>

      <section className="metric-grid">
        <Metric
          label="ACCURACY"
          value={report.metrics.accuracy}
        />

        <Metric
          label="PRECISION"
          value={report.metrics.precision}
        />

        <Metric
          label="RECALL"
          value={report.metrics.recall}
        />

        <Metric
          label="F1"
          value={report.metrics.f1}
        />

        {report.metrics.roc_auc !==
          undefined && (
          <Metric
            label="ROC-AUC"
            value={report.metrics.roc_auc}
          />
        )}
      </section>

      <section className="content-grid">
        <div className="panel findings">
          <PanelTitle
            icon={
              <ShieldAlert size={17} />
            }
            title="FORENSIC FINDINGS"
          />

          {report.findings.length ===
          0 ? (
            <div className="empty">
              No major findings detected.
            </div>
          ) : (
            report.findings.map(
              (finding, index) => (
                <div
                  className="finding"
                  key={index}
                >
                  <div
                    className={`severity ${finding.severity}`}
                  >
                    {finding.severity}
                  </div>

                  <div>
                    <strong>
                      {finding.title}
                    </strong>

                    <p>
                      {finding.description}
                    </p>
                  </div>
                </div>
              )
            )
          )}
        </div>

        <div className="panel">
          <PanelTitle
            icon={
              <BrainCircuit size={17} />
            }
            title="FEATURE EVIDENCE"
          />

          {report.feature_importance.features
            .slice(0, 7)
            .map((item) => {
              const max =
                report.feature_importance
                  .features[0]
                  ?.importance || 1;

              return (
                <div
                  className="feature-row"
                  key={item.feature}
                >
                  <div>
                    <span>
                      {item.feature}
                    </span>

                    <b>
                      {item.importance.toFixed(
                        3
                      )}
                    </b>
                  </div>

                  <div className="bar">
                    <span
                      style={{
                        width: `${
                          (item.importance /
                            max) *
                          100
                        }%`,
                      }}
                    />
                  </div>
                </div>
              );
            })}
        </div>
      </section>

      <section className="content-grid">
        <div className="panel">
          <PanelTitle
            icon={<Target size={17} />}
            title="REPRESENTATIVE CASE"
          />

          <div className="case-header">
            <span>
              ROW #{report.representative_case.index}
            </span>

            <strong>
              PREDICTION:{" "}
              {
                report
                  .representative_case
                  .prediction
              }
            </strong>
          </div>

          {report.representative_case
            .contributions
            .slice(0, 6)
            .map((item) => (
              <div
                className="contribution"
                key={item.feature}
              >
                <span>
                  {item.feature}
                </span>

                <span>
                  {String(item.value)}
                </span>

                <b
                  className={
                    item.contribution >= 0
                      ? "positive"
                      : "negative"
                  }
                >
                  {item.contribution >= 0
                    ? "+"
                    : ""}
                  {item.contribution.toFixed(
                    3
                  )}
                </b>
              </div>
            ))}
        </div>

        <div className="panel counterfactual">
          <PanelTitle
            icon={<Target size={17} />}
            title="COUNTERFACTUAL"
          />

          {report.counterfactual
            .found ? (
            <>
              <p>
                A prediction flip was
                found by changing{" "}
                <strong>
                  {
                    report.counterfactual
                      .changes?.length
                  }
                </strong>{" "}
                feature(s).
              </p>

              {report.counterfactual.changes?.map(
                (change) => (
                  <div
                    className="change"
                    key={change.feature}
                  >
                    <span>
                      {change.feature}
                    </span>

                    <strong>
                      {String(change.from)}
                      {" → "}
                      {String(change.to)}
                    </strong>
                  </div>
                )
              )}
            </>
          ) : (
            <div className="empty">
              No actionable prediction
              flip was found within the
              search budget.
            </div>
          )}
        </div>
      </section>
    </main>
  );
}

function Metric({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>
        {(value * 100).toFixed(1)}
        <small>%</small>
      </strong>
    </div>
  );
}

function PanelTitle({
  icon,
  title,
}: {
  icon: React.ReactNode;
  title: string;
}) {
  return (
    <div className="panel-title">
      {icon}
      <span>{title}</span>
    </div>
  );
}

export default App;