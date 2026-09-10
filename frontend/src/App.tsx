import {
  useEffect,
  useState,
} from "react";

import Login from "./components/Login";
import UploadWorkspace from "./components/UploadWorkspace";
import OutputReport from "./components/OutputReport";

import type { Report } from "./components/types";

import "./styles.css";

function App() {
  const [
    authenticated,
    setAuthenticated,
  ] = useState(false);

  const [
    report,
    setReport,
  ] = useState<Report | null>(null);

  useEffect(() => {
    setAuthenticated(
      localStorage.getItem(
        "model-autopsy-auth",
      ) === "true",
    );
  }, []);

  const handleLogin = () => {
    localStorage.setItem(
      "model-autopsy-auth",
      "true",
    );

    setAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem(
      "model-autopsy-auth",
    );

    setAuthenticated(false);
    setReport(null);
  };

  const handleNewAutopsy = () => {
    setReport(null);

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  if (!authenticated) {
    return (
      <Login
        onLogin={handleLogin}
      />
    );
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
      <header className="topbar report-nav">
        <button
          className="brand"
          type="button"
          onClick={handleNewAutopsy}
        >
          <span className="brand-mark">
            MA
          </span>

          <span className="brand-text">
            <strong>MODEL</strong>
            <span>AUTOPSY</span>
          </span>
        </button>

        <div className="workspace-actions">
          <div className="engine-status">
            <i />
            FORENSIC ENGINE ONLINE
          </div>

          <button
            className="secondary-button"
            type="button"
            onClick={handleNewAutopsy}
          >
            NEW AUTOPSY
          </button>

          <button
            className="logout-button"
            type="button"
            onClick={handleLogout}
          >
            SIGN OUT
          </button>
        </div>
      </header>

      <main>
        <OutputReport
          report={report}
        />

        <footer className="report-footer">
          <div>
            <div className="footer-brand">
              MODEL AUTOPSY
            </div>

            <p>
              Forensic debugging and
              explainability for machine
              learning models.
            </p>
          </div>

          <button
            className="secondary-button"
            type="button"
            onClick={
              handleNewAutopsy
            }
          >
            RUN ANOTHER AUTOPSY
          </button>
        </footer>
      </main>
    </div>
  );
}

export default App;