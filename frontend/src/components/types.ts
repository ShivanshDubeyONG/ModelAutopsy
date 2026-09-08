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
  lift?: number;
  size?: number;
  [key: string]: unknown;
};

export type Metrics = {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  roc_auc?: number;
  confusion_matrix: number[][];
  labels: Array<string | number>;
};

export type Report = {
  model: {
    name: string;
    problem_type: string;
  };

  dataset: {
    samples: number;
    features: number;
    feature_names: string[];
    target: string;
  };

  health_score: number;

  metrics: Metrics;

  error_analysis: ErrorSlice[];

  feature_importance: {
    method: string;
    features: FeatureImportance[];
    error?: string;
  };

  representative_case: {
    index: number;
    prediction: number;
    probability: number | null;
    method: string;
    contributions: Contribution[];
  };

  counterfactual: {
    found: boolean;
    original_prediction: number;
    desired_prediction: number;
    new_prediction?: number;

    changes?: Array<{
      feature: string;
      from: string | number;
      to: string | number;
    }>;

    n_features_changed?: number;
  };

  findings: Finding[];
};