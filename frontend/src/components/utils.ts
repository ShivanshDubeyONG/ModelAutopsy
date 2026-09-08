export function prettyName(
  value: string | undefined,
): string {
  if (!value) return "";

  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) =>
      letter.toUpperCase(),
    );
}

export function formatPercent(
  value: number | undefined,
  digits = 1,
): string {
  if (
    value === undefined ||
    value === null ||
    Number.isNaN(value)
  ) {
    return "—";
  }

  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(
  value: number | undefined,
  digits = 3,
): string {
  if (
    value === undefined ||
    value === null ||
    Number.isNaN(value)
  ) {
    return "—";
  }

  return value.toFixed(digits);
}

export function healthLabel(score: number): string {
  if (score >= 80) return "Healthy";
  if (score >= 65) return "Watch";
  if (score >= 45) return "At Risk";
  return "Critical";
}

export function healthClass(score: number): string {
  if (score >= 80) return "healthy";
  if (score >= 65) return "watch";
  if (score >= 45) return "risk";
  return "critical";
}

export function severityClass(
  severity: string,
): string {
  const value = severity.toLowerCase();

  if (value === "critical") return "critical";
  if (value === "warning") return "warning";

  return "info";
}