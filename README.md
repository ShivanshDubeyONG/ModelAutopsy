# Model Autopsy

Model Autopsy investigates trained machine-learning models on
unseen labeled data and turns model failures into evidence.

Instead of stopping at:

"Your model has 93% accuracy."

Model Autopsy asks:

- Where does the model fail?
- Which data cohorts are disproportionately misclassified?
- Which features drive its decisions?
- Why did a particular prediction happen?
- What evidence supports each finding?

## How it works

```text
Trained Model + Evaluation Dataset
                |
                v
         Model Autopsy Engine
                |
        +-------+-------+
        |       |       |
     Metrics  Errors  Explainability
        |       |       |
        +-------+-------+
                |
                v
        Evidence Chain
                |
                v
       Interactive Report
