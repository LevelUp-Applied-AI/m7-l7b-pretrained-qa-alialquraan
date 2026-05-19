Hypothesis

The QA model is expected to fail in contexts where multiple entities of the same type appear simultaneously, especially when distractor entities are positioned near the correct answer. The input pattern is entity-rich contexts with same-type distractors that compete for selection.

The expected output failure is incorrect span selection of a plausible distractor entity that matches the question’s entity type but not its relational role.

I hypothesized that the model relies heavily on lexical overlap and entity salience rather than deep relational reasoning between the question and the specific entity role in context.

Set Design

The adversarial dataset contains 30 examples divided into three pattern tags:

same_type_distractor (10 examples): contexts containing multiple entities of the same type where only one is correct.

nearby_distractor (5 examples): distractors placed close to the correct entity to test local confusion effects.

control_easy (15 examples): simplified contexts without distractors to verify whether errors are due to difficulty or distractor interference.

Control examples are essential because they isolate the effect of distractors. High performance on control cases indicates that the model understands the task but may struggle with ambiguity in entity selection.

Results

Overall performance remains strong:

Overall EM: 0.9333  
Overall F1: 0.9333  
n: 30

Per-pattern observations:

same_type_distractor:
This pattern shows slightly lower performance compared to control cases, indicating mild sensitivity to entity competition.

Example:
qid: EX_01  
question: Which company acquired Instagram?  
gold: Facebook  
prediction: Facebook  

control_easy:
The model performs near perfectly on simplified inputs, confirming that the base QA capability is strong.

Example:
qid: EX_02  
question: Which company created Windows?  
gold: Microsoft  
prediction: Microsoft  

nearby_distractor:
Performance remains stable, suggesting limited but not severe degradation under local distractor pressure.

Example:
qid: EX_11  
question: Who directed Inception?  
gold: Christopher Nolan  
prediction: Christopher Nolan  

Production Defense

A practical and effective mitigation is confidence-based abstention.

If the QA model’s prediction confidence falls below a calibrated threshold, the system should abstain and either:
- route the query to a human reviewer, or
- trigger a retrieval-augmented verification step

This approach directly targets ambiguity cases caused by distractor entities without degrading performance on clean inputs, making it suitable for production systems where precision is critical.