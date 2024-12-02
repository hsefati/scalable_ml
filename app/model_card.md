# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

- Model Name: Salary Prediction Model
- Version: 1.0
- Framework Used: Scikit-learn
- Model Type: Logistic Regression
- Preprocessing Pipeline:
  - Numerical Features: StandardScaler
  - Categorical Features: OneHotEncoder
- Input Features:
  - Numerical: age, fnlgt, education-num, capital-gain, capital-loss, hours-per-week
  - Categorical: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Target Variable: salary (binary classification: \<=50K or >50K)

## Intended Use

- Primary Purpose: To predict whether a person's annual income exceeds $50,000 (>50K) based on demographic and employment attributes.
- Intended Users:
  - Researchers studying factors influencing income distribution.
- Use Cases:
  - Assessing income inequality trends.

## Training Data

- Source: Simulated data based on the Adult Census Income dataset.
- Attributes:
  - Demographic features: age, race, sex, relationship.
  - Employment-related features: workclass, occupation, education.

## Evaluation Data

- Source: A reserved test split (20% of the dataset).
- Attributes:
  - Same as the training dataset.
  - Ensures evaluation on unseen data for unbiased performance estimation.

## Metrics

- Metrics Used:
  - Precision: Proportion of true positives among predicted positives.
  - Recall: Proportion of true positives among all actual positives.
  - F1-Score: Harmonic mean of precision and recall.
- Model Performance on Test Data:
  - Precision: 96%
  - Recall: 92%
  - F1-Score: 94%

## Ethical Considerations

- Bias in Data:
  - Model predictions depend on demographic and employment data, which may reflect societal biases (e.g., gender or race-related disparities).
    addressing potential biases.
- Privacy:
  - Data used for training and evaluation should comply with privacy regulations (e.g., anonymization, data minimization).

## Caveats and Recommendations

- Recommendations:
  - Before deployment, train the model on a complete and diverse dataset.
  - Regularly audit the model for fairness, bias, and performance degradation.
  - Avoid using the model for critical decisions without expert oversight.
