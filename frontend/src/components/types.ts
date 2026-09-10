export type FeatureImportance = {
  feature: string;
  importance: number;
};

export type Contribution = {
  feature: string;
  value: string | number;
  contribution: number;
};

export type Finding = {
  severity: string;
  type: string;
  title: string;
  description: string;

  evidence?: Array<{
    feature?: string;
    importance?: number;
    [key: string]: unknown;
  }>;

  score?: number;
};

export type ErrorSlice = {
  feature?: string;
  value?: string | number;

  error_rate?: number;
  baseline_error?: number;
  error?: number;

  lift?: number;
  size?: number;

  error_metric?: string;
  problem_type?: string;

  [key: string]: unknown;
};

export type Metrics = {
  problem_type?: string;

  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  roc_auc?: number;

  mae?: number;
  rmse?: number;
  r2?: number;
  mape?: number;

  confusion_matrix?: number[][];
  labels?: Array<string | number>;

  [key: string]: unknown;
};

export type FeatureEvidence = {
  method?: string;
  features?: FeatureImportance[];
  error?: string;
  [key: string]: unknown;
};

export type RepresentativeCase = {
  index: number;

  actual?: unknown;

  prediction: unknown;

  probability?: number | null;

  probability_label?: unknown;

  method?: string;

  features?: Record<
    string,
    number
  >;

  contributions?: Array<{
    feature: string;
    value: string | number | boolean | null;
    contribution: number;
  }>;

  [key: string]: unknown;
};

export interface Counterfactual {
  found: boolean;

  original_prediction?: string | number;
  final_prediction?: string | number;
  desired_prediction?: string | number;

  desired_probability?: number;

  changes: {
    feature: string;
    from: string | number;
    to: string | number;
    desired_probability?: number;
  }[];

  reason?: string;
}

export type OutputReport = {
  name: string;
  problem_type: string;

  health_score: number;

  metrics: Metrics;

  error_analysis: ErrorSlice[];

  feature_importance: FeatureEvidence;

  representative_case: RepresentativeCase;

  counterfactual: Counterfactual;

  findings: Finding[];
};

export type Report = {
  model: {
    name: string;
    problem_type: string;
    n_outputs?: number;
  };

  dataset: {
    samples: number;
    features: number;
    feature_names: string[];

    target?: string;
    targets?: string[];
  };

  health_score: number;

  outputs?: OutputReport[];

  // Legacy single-output compatibility.
  metrics?: Metrics;
  error_analysis?: ErrorSlice[];
  feature_importance?: FeatureEvidence;
  representative_case?: RepresentativeCase;
  counterfactual?: Counterfactual;
  findings?: Finding[];
};