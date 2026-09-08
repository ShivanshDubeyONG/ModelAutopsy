import {
  ArrowRight,
  GitBranch,
  Zap,
} from "lucide-react";

import type { Report } from "./types";
import { prettyName } from "./utils";

type Props = {
  data: Report["counterfactual"];
};

export default function Counterfactual({
  data,
}: Props) {
  return (
    <section className="report-section">
      <div className="section-heading">
        <div className="section-heading-left">
          <span className="section-number">
            05
          </span>

          <div>
            <div className="section-eyebrow">
              <GitBranch size={13} />
              COUNTERFACTUAL
            </div>

            <h2>
              What would change the decision?
            </h2>

            <p>
              The smallest discovered feature
              change capable of flipping the
              prediction.
            </p>
          </div>
        </div>
      </div>

      {data.found ? (
        <div className="counterfactual-card">
          <div className="cf-predictions">
            <div>
              <span>ORIGINAL</span>
              <strong>
                {data.original_prediction}
              </strong>
            </div>

            <ArrowRight
              className="cf-arrow"
              size={19}
            />

            <div className="cf-target">
              <span>FLIPPED TO</span>
              <strong>
                {data.new_prediction ??
                  data.desired_prediction}
              </strong>
            </div>
          </div>

          <div className="cf-divider" />

          <div className="cf-heading">
            <div>
              <span className="micro-label">
                MINIMAL CHANGE
              </span>

              <h3>
                {data.n_features_changed ??
                  0}{" "}
                feature
                {(data.n_features_changed ??
                  0) === 1
                  ? ""
                  : "s"}{" "}
                changed
              </h3>
            </div>

            <Zap
              size={19}
            />
          </div>

          {data.changes?.map(
            (change) => (
              <div
                className="cf-change"
                key={change.feature}
              >
                <strong>
                  {prettyName(
                    change.feature,
                  )}
                </strong>

                <div>
                  <span>FROM</span>
                  <b>
                    {String(
                      change.from,
                    )}
                  </b>
                </div>

                <ArrowRight
                  size={15}
                />

                <div className="cf-new">
                  <span>TO</span>
                  <b>
                    {String(
                      change.to,
                    )}
                  </b>
                </div>
              </div>
            ),
          )}
        </div>
      ) : (
        <div className="empty-state">
          <GitBranch size={17} />

          <div>
            <strong>
              No prediction-flipping
              counterfactual found.
            </strong>

            <p>
              The explored candidate space did
              not produce a valid feature change.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}