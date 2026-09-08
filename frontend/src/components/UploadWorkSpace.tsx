import { useRef, useState } from "react";
import {
  ArrowUpRight,
  Check,
  FileText,
  Upload,
  X,
} from "lucide-react";

import type { Report } from "./types";

type UploadWorkspaceProps = {
  onReport: (report: Report) => void;
  onLogout: () => void;
};

function UploadWorkspace({
  onReport,
  onLogout,
}: UploadWorkspaceProps) {
  const modelInputRef = useRef<HTMLInputElement>(null);
  const datasetInputRef = useRef<HTMLInputElement>(null);

  const [modelFile, setModelFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("approved");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runAutopsy = async () => {
    if (!modelFile || !datasetFile || !targetColumn.trim()) {
      setError("Upload a model, evaluation dataset, and target column.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();

      formData.append("model", modelFile);
      formData.append("dataset", datasetFile);
      formData.append("target_column", targetColumn.trim());

      const response = await fetch("http://127.0.0.1:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.error ||
            data?.detail ||
            "The analysis request failed.",
        );
      }

      if (!data.success || !data.report) {
        throw new Error(
          data?.error || "The backend returned no analysis report.",
        );
      }

      onReport(data.report as Report);
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

  const handleFile = (
    file: File | undefined,
    type: "model" | "dataset",
  ) => {
    if (!file) return;

    if (type === "model") {
      const valid =
        file.name.toLowerCase().endsWith(".joblib") ||
        file.name.toLowerCase().endsWith(".pkl");

      if (!valid) {
        setError("Model must be a .joblib or .pkl file.");
        return;
      }

      setModelFile(file);
      setError("");
      return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Evaluation dataset must be a .csv file.");
      return;
    }

    setDatasetFile(file);
    setError("");
  };

  const handleDrop = (
    event: React.DragEvent<HTMLDivElement>,
    type: "model" | "dataset",
  ) => {
    event.preventDefault();
    handleFile(event.dataTransfer.files?.[0], type);
  };

  return (
    <main className="workspace-page">
      <div className="workspace-layout">
        <section className="workspace-intro">
          <p className="eyebrow">FORENSIC ML ANALYSIS</p>

          <h1>
            Find out
            <br />
            <span>why your model</span>
            <br />
            fails.
          </h1>

          <p className="workspace-description">
            Model Autopsy investigates a trained machine learning
            model against unseen labeled data, surfacing the evidence
            behind its mistakes.
          </p>

          <div className="workspace-steps">
            <div className="workspace-step">
              <FileText size={19} />
              <span>MODEL</span>
            </div>

            <div className="workspace-arrow">›</div>

            <div className="workspace-step">
              <FileText size={19} />
              <span>EVIDENCE</span>
            </div>

            <div className="workspace-arrow">›</div>

            <div className="workspace-step workspace-step-active">
              <Check size={19} />
              <span>AUTOPSY</span>
            </div>
          </div>
        </section>

        <section className="upload-workspace">
          <div className="workspace-heading">
            <div>
              <p className="eyebrow">START INVESTIGATION</p>
              <h2>Open an autopsy.</h2>
            </div>

            <div className="workspace-icon">
              <Upload size={20} strokeWidth={1.7} />
            </div>
          </div>

          <div className="upload-divider" />

          <div
            className="upload-card"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => handleDrop(event, "model")}
          >
            <input
              ref={modelInputRef}
              type="file"
              accept=".joblib,.pkl"
              hidden
              onChange={(event) =>
                handleFile(event.target.files?.[0], "model")
              }
            />

            <div className="upload-card-icon">
              {modelFile ? (
                <Check size={18} />
              ) : (
                <Upload size={18} />
              )}
            </div>

            <div className="upload-card-content">
              <strong>
                {modelFile ? modelFile.name : "Model artifact"}
              </strong>

              <span>
                {modelFile
                  ? "Model ready for analysis"
                  : ".joblib or .pkl"}
              </span>
            </div>

            {modelFile ? (
              <button
                className="file-remove"
                type="button"
                onClick={() => setModelFile(null)}
                aria-label="Remove model"
              >
                <X size={16} />
              </button>
            ) : (
              <button
                className="file-action"
                type="button"
                onClick={() => modelInputRef.current?.click()}
              >
                Choose
              </button>
            )}
          </div>

          <div
            className="upload-card"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => handleDrop(event, "dataset")}
          >
            <input
              ref={datasetInputRef}
              type="file"
              accept=".csv"
              hidden
              onChange={(event) =>
                handleFile(event.target.files?.[0], "dataset")
              }
            />

            <div className="upload-card-icon">
              {datasetFile ? (
                <Check size={18} />
              ) : (
                <FileText size={18} />
              )}
            </div>

            <div className="upload-card-content">
              <strong>
                {datasetFile ? datasetFile.name : "Evaluation dataset"}
              </strong>

              <span>
                {datasetFile
                  ? "Evaluation set ready"
                  : "Labeled .csv file"}
              </span>
            </div>

            {datasetFile ? (
              <button
                className="file-remove"
                type="button"
                onClick={() => setDatasetFile(null)}
                aria-label="Remove dataset"
              >
                <X size={16} />
              </button>
            ) : (
              <button
                className="file-action"
                type="button"
                onClick={() => datasetInputRef.current?.click()}
              >
                Choose
              </button>
            )}
          </div>

          <label className="target-field">
            <span>TARGET COLUMN</span>

            <input
              type="text"
              value={targetColumn}
              onChange={(event) =>
                setTargetColumn(event.target.value)
              }
              placeholder="e.g. approved"
            />
          </label>

          {error && (
            <div className="workspace-error">
              {error}
            </div>
          )}

          <button
            className="run-autopsy-button"
            type="button"
            disabled={
              !modelFile ||
              !datasetFile ||
              !targetColumn.trim() ||
              loading
            }
            onClick={runAutopsy}
          >
            <span>
              {loading ? "ANALYZING..." : "RUN AUTOPSY"}
            </span>

            <ArrowUpRight size={18} />
          </button>

          <div className="workspace-meta">
            <span>PRIVATE</span>
            <span>LOCAL</span>
            <span>NO TRAINING DATA REQUIRED</span>
          </div>

          <button
            className="workspace-logout"
            type="button"
            onClick={onLogout}
          >
            Sign out
          </button>
        </section>
      </div>
    </main>
  );
}

export default UploadWorkspace;