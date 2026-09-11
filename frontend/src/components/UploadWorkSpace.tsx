import { useRef, useState } from "react";

import {
  ArrowUpRight,
  Check,
  FileBox,
  FileSpreadsheet,
  Fingerprint,
  X,
  Zap,
} from "lucide-react";

type UploadWorkspaceProps = {
  onReport: (report: any) => void;
  onLogout: () => void;
};

function UploadWorkspace({
  onReport,
  onLogout,
}: UploadWorkspaceProps) {
  const modelInputRef =
    useRef<HTMLInputElement>(null);

  const datasetInputRef =
    useRef<HTMLInputElement>(null);

  const [modelFile, setModelFile] =
    useState<File | null>(null);

  const [datasetFile, setDatasetFile] =
    useState<File | null>(null);

  const [targetColumn, setTargetColumn] =
    useState("approved");

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState("");

  const targets = targetColumn
    .split(",")
    .map((target) => target.trim())
    .filter(Boolean);

  const runAutopsy = async () => {
    if (
      !modelFile ||
      !datasetFile ||
      !targetColumn.trim()
    ) {
      setError(
        "Upload a model, evaluation dataset, and at least one target output.",
      );

      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append(
        "model",
        modelFile,
      );

      formData.append(
        "dataset",
        datasetFile,
      );

      formData.append(
        "target_column",
        targetColumn.trim(),
      );

      const response = await fetch(
        "http://127.0.0.1:8000/api/analyze",
        {
          method: "POST",
          body: formData,
        },
      );

      const result =
        await response.json();
        console.log("AUTOPSY RESPONSE:", result);

      if (
        !response.ok ||
        !result.success
      ) {
        throw new Error(
          result.error ||
            "Autopsy analysis failed.",
        );
      }

      onReport(result.report);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Something went wrong while running the autopsy.",
      );
    } finally {
      setLoading(false);
    }
  };

  const removeModel = () => {
    setModelFile(null);

    if (modelInputRef.current) {
      modelInputRef.current.value = "";
    }
  };

  const removeDataset = () => {
    setDatasetFile(null);

    if (datasetInputRef.current) {
      datasetInputRef.current.value = "";
    }
  };

  return (
    <div className="workspace-app">
      <header className="topbar">
        <button
          className="brand"
          type="button"
          onClick={() =>
            window.location.reload()
          }
          aria-label="Model Autopsy home"
        >
          <span className="brand-mark">
            <Fingerprint
              size={20}
              strokeWidth={1.5}
            />
          </span>

          <span className="brand-text">
            <strong>MODEL</strong>
            <span>AUTOPSY</span>
          </span>
        </button>

        <div className="workspace-actions">
          <div className="engine-status">
            <i />
            LOCAL FORENSIC ENGINE
          </div>

          <button
            className="logout-button"
            type="button"
            onClick={onLogout}
          >
            SIGN OUT
          </button>
        </div>
      </header>

      <main className="workspace">
        <section className="workspace-heading">
          <div>
            <div className="eyebrow">
              <Fingerprint size={12} />
              FORENSIC ML ANALYSIS
            </div>

            <h1>
              Find out
              <br />
              <span className="hero-accent">
                why your model
              </span>
              <br />
              fails.
            </h1>

            <p>
              Your model made the mistake.
              <span className="hero-description-accent">
                {" "}
                We find the evidence.
              </span>
              <br />
              Trace failure cohorts, feature
              drivers, representative cases,
              and model behavior.
            </p>
          </div>

          <Zap size={24} />
        </section>

        <section className="workspace-grid">
          <div className="workspace-card">
            <div className="workspace-card-heading">
              <span className="step-number">
                01
              </span>

              <div>
                <div className="micro-label">
                  <FileBox size={12} />
                  MODEL ARTIFACT
                </div>

                <h2>
                  Upload the trained model.
                </h2>

                <p>
                  Serialized scikit-learn
                  compatible model.
                </p>
              </div>
            </div>

            <label className="workspace-dropzone">
              <input
                ref={modelInputRef}
                type="file"
                accept=".joblib,.pkl"
                onChange={(event) =>
                  setModelFile(
                    event.target.files?.[0] ??
                      null,
                  )
                }
              />

              <div className="drop-icon">
                {modelFile ? (
                  <Check size={18} />
                ) : (
                  <FileBox size={18} />
                )}
              </div>

              <div className="drop-copy">
                <strong>
                  {modelFile
                    ? modelFile.name
                    : "Choose model artifact"}
                </strong>

                <span>
                  {modelFile
                    ? "Model ready for analysis"
                    : ".joblib or .pkl"}
                </span>
              </div>

              {modelFile && (
                <button
                  className="remove-file"
                  type="button"
                  aria-label="Remove model"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    removeModel();
                  }}
                >
                  <X size={15} />
                </button>
              )}
            </label>
          </div>

          <div className="workspace-card">
            <div className="workspace-card-heading">
              <span className="step-number">
                02
              </span>

              <div>
                <div className="micro-label">
                  <FileSpreadsheet
                    size={12}
                  />
                  EVIDENCE DATASET
                </div>

                <h2>
                  Upload evaluation data.
                </h2>

                <p>
                  Unseen data used to expose
                  model failure patterns.
                </p>
              </div>
            </div>

            <label className="workspace-dropzone">
              <input
                ref={datasetInputRef}
                type="file"
                accept=".csv"
                onChange={(event) =>
                  setDatasetFile(
                    event.target.files?.[0] ??
                      null,
                  )
                }
              />

              <div className="drop-icon">
                {datasetFile ? (
                  <Check size={18} />
                ) : (
                  <FileSpreadsheet
                    size={18}
                  />
                )}
              </div>

              <div className="drop-copy">
                <strong>
                  {datasetFile
                    ? datasetFile.name
                    : "Choose evaluation dataset"}
                </strong>

                <span>
                  {datasetFile
                    ? "Evaluation set ready"
                    : "Labeled .csv file"}
                </span>
              </div>

              {datasetFile && (
                <button
                  className="remove-file"
                  type="button"
                  aria-label="Remove dataset"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    removeDataset();
                  }}
                >
                  <X size={15} />
                </button>
              )}
            </label>
          </div>
        </section>

        <section className="configuration-card autopsy-config">
          <div>
            <div className="micro-label">
              <Fingerprint size={12} />
              INVESTIGATION TARGETS
            </div>

            <h2>
              What outputs should we autopsy?
            </h2>

            <p className="config-description">
              Enter one target or multiple
              comma-separated outputs.
            </p>
          </div>

          <div className="target-control">
            <label htmlFor="target-column">
              TARGET OUTPUT(S)
            </label>

            <input
              id="target-column"
              type="text"
              value={targetColumn}
              onChange={(event) =>
                setTargetColumn(
                  event.target.value,
                )
              }
              placeholder="Reference_Parameter, Validity_Label"
            />

            {targets.length > 0 && (
              <div className="target-chips">
                {targets.map((target) => (
                  <span key={target}>
                    {target}
                  </span>
                ))}
              </div>
            )}
          </div>
        </section>

        {error && (
          <div className="workspace-error">
            {error}
          </div>
        )}

        <div className="workspace-footer">
          <div className="workspace-meta">
            <span>PRIVATE</span>
            <span>LOCAL</span>
            <span>
              MULTI-OUTPUT READY
            </span>
          </div>

          <button
            className="primary-button workspace-run"
            type="button"
            disabled={
              loading ||
              !modelFile ||
              !datasetFile ||
              !targetColumn.trim()
            }
            onClick={runAutopsy}
          >
            {loading ? (
              "RUNNING AUTOPSY..."
            ) : (
              <>
                RUN AUTOPSY
                <ArrowUpRight size={15} />
              </>
            )}
          </button>
        </div>
      </main>
    </div>
  );
}

export default UploadWorkspace;