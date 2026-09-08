import {
  Activity,
  ArrowRight,
  LockKeyhole,
  Mail,
} from "lucide-react";
import { useState } from "react";

type Props = {
  onLogin: () => void;
};

export default function Login({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [password, setPassword] =
    useState("");
  const [error, setError] = useState("");

  const submit = (
    event: React.FormEvent,
  ) => {
    event.preventDefault();

    if (!email.trim() || !password) {
      setError(
        "Enter your email and password.",
      );
      return;
    }

    setError("");
    onLogin();
  };

  return (
    <div className="auth-page">
      <div className="auth-shell">
        <div className="auth-brand">
          <div className="brand-mark">
            <Activity size={21} />
          </div>

          <div className="brand-text">
            <strong>MODEL</strong>
            <span>AUTOPSY</span>
          </div>
        </div>

        <div className="auth-card">
          <div className="auth-heading">
            <span className="eyebrow">
              SECURE WORKSPACE
            </span>

            <h1>Welcome back.</h1>

            <p>
              Sign in to continue your model
              investigations.
            </p>
          </div>

          <form onSubmit={submit}>
            <label className="auth-field">
              <span>
                <Mail size={13} />
                EMAIL
              </span>

              <input
                type="email"
                placeholder="you@example.com"
                value={email}
                onChange={(event) =>
                  setEmail(event.target.value)
                }
                autoComplete="email"
              />
            </label>

            <label className="auth-field">
              <span>
                <LockKeyhole size={13} />
                PASSWORD
              </span>

              <input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) =>
                  setPassword(
                    event.target.value,
                  )
                }
                autoComplete="current-password"
              />
            </label>

            {error && (
              <div className="auth-error">
                {error}
              </div>
            )}

            <button
              className="primary-button"
              type="submit"
            >
              SIGN IN
              <ArrowRight size={16} />
            </button>
          </form>

          <div className="auth-note">
            MODEL AUTOPSY · PRIVATE ML ANALYSIS
          </div>
        </div>
      </div>
    </div>
  );
}