hypothesis
The QA model underperforms significantly when multiple similar entities appear within the same context and when exact token span boundaries are strictly required. It struggles with subtle context shifts.

set design
We evaluate the model performance by sorting predictions in qa_predictions.csv by ascending F1 score and manually inspecting the lowest-performing examples to identify recurring error patterns.

results
We observe three main failure patterns: partial span extraction issues, distractor entity confusion, and punctuation-related prediction noise.

production defense
The model should not be deployed in high-stakes applications such as legal documentation review or medical QA. It lacks uncertainty estimation, abstention capability, and robustness to distractor-heavy contexts.