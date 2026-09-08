import { Activity } from "lucide-react";

type Props = {
  onClick?: () => void;
};

export default function Brand({ onClick }: Props) {
  return (
    <button
      className="brand"
      onClick={onClick}
      aria-label="Refresh Model Autopsy"
    >
      <div className="brand-mark">
        <Activity size={19} />
      </div>

      <div className="brand-text">
        <strong>MODEL</strong>
        <span>AUTOPSY</span>
      </div>
    </button>
  );
}