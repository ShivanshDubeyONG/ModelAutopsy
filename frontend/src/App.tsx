import { useEffect, useState } from "react";

import Login from "./components/Login";
import UploadWorkspace from "./components/UploadWorkspace";
import ReportHeader from "./components/ReportHeader";
import MetricsGrid from "./components/MetricsGrid";
import Findings from "./components/Findings";
import FeatureEvidence from "./components/FeatureEvidence";
import ErrorAnalysis from "./components/ErrorAnalysis";
import RepresentativeCase from "./components/RepresentativeCase";
import Counterfactual from "./components/Counterfactual";
import ConfusionMatrix from "./components/ConfusionMatrix";

import type { Report } from "./components/types";
import "./styles.css";

function App() {
  const [authenticated, setAuthenticated] = useState(false);
  const [report, setReport] = useState<Report | null>(null);

  useEffect(() => {
    setAuthenticated(
      localStorage.getItem("model-autopsy-auth") === "true",
    );
  }, []);

  const handleLogin = () => {
    localStorage.setItem("model-autopsy-auth", "true");
    setAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("model-autopsy-auth");
    setAuthenticated(false);
    setReport(null);
  };

  const handleNewAutopsy = () => {
    setReport(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  if (!authenticated) {
    return <Login onLogin={handleLogin} />;
  }

  if (!report) {
    return (
      <UploadWorkspace
        onReport={setReport}
        onLogout={handleLogout}
      />
    );
  }

  return (
    <div className="app">
      <ReportHeader
        report={report}
        onNewAutopsy={handleNewAutopsy}
        onLogout={handleLogout}
      />

      <main className="report">
        <MetricsGrid metrics={report.metrics} />

        <Findings findings={report.findings} />

        <FeatureEvidence
          report={report}
        />

        <ErrorAnalysis
          slices={report.error_analysis}
        />

        <RepresentativeCase
          caseData={report.representative_case}
        />

        <Counterfactual
          data={report.counterfactual}
        />

        <ConfusionMatrix
          metrics={report.metrics}
        />

        <footer className="report-footer">
          <div>
            <div className="footer-brand">
              MODEL AUTOPSY
            </div>

            <p>
              Forensic debugging and explainability
              for machine learning models.
            </p>
          </div>

          <button
            className="secondary-button"
            onClick={handleNewAutopsy}
          >
            RUN ANOTHER AUTOPSY
          </button>
        </footer>
      </main>
    </div>
  );
}

export default App;