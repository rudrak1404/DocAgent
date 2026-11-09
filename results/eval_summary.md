# FinanceBench evaluation (150 questions)

Graded: baseline 150/150, agent 150/150

Evidence on gold page: DocAgent = its cited evidence; baseline = its 5 retrieved passages.

| system | subset | n | accuracy | incorrect | refusal | acc. when answered | faithfulness | evidence on gold page | $/question | s/question | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | all | 150 | **34.0%** | 44 | 55 | 53.7% | 96.2% | 41.3% | $0.000 | 110 | 0.0 |
| baseline | test split | 75 | **32.0%** | 24 | 27 | 50.0% | 95.3% | 42.7% | $0.000 | 116 | 0.0 |
| baseline | excluding pilot | 140 | **34.3%** | 40 | 52 | 54.5% | 96.7% | 39.3% | $0.000 | 110 | 0.0 |
| baseline | metrics-generated | 50 | **32.0%** | 15 | 19 | 51.6% | 100.0% | 38.0% | $0.000 | 104 | 0.0 |
| baseline | domain-relevant | 50 | **34.0%** | 18 | 15 | 48.6% | 93.6% | 38.0% | $0.000 | 102 | 0.0 |
| baseline | novel-generated | 50 | **36.0%** | 11 | 21 | 62.1% | 95.4% | 48.0% | $0.000 | 124 | 0.0 |
| agent | all | 150 | **52.7%** | 40 | 31 | 66.4% | 92.3% | 63.4% | $0.000 | 88 | 5.7 |
| agent | test split | 75 | **53.3%** | 21 | 14 | 65.6% | 91.7% | 69.0% | $0.000 | 87 | 5.6 |
| agent | excluding pilot | 140 | **52.9%** | 39 | 27 | 65.5% | 91.9% | 62.9% | $0.000 | 86 | 5.5 |
| agent | metrics-generated | 50 | **68.0%** | 11 | 5 | 75.6% | 97.6% | 71.4% | $0.000 | 70 | 5.3 |
| agent | domain-relevant | 50 | **44.0%** | 13 | 15 | 62.9% | 90.7% | 61.7% | $0.000 | 103 | 6.3 |
| agent | novel-generated | 50 | **46.0%** | 16 | 11 | 59.0% | 87.6% | 56.5% | $0.000 | 92 | 5.4 |

## Paired comparison (same questions)

DocAgent correct where the baseline is not: 40; the reverse: 12. Exact McNemar test p = 0.00013. Accuracy difference +18.7% (95% bootstrap CI +9.3% to +28.0%).

## Guardrails

Withheld answers blocked by the guardrails: 16. Graded anyway: 11 would have been correct, 5 incorrect, 0 non-answers.
Withheld answers flagged by the agent as insufficient evidence: 15. Graded anyway: 3 would have been correct, 5 incorrect, 7 non-answers.
If every withheld draft had been returned instead: accuracy 62.0% with 50 incorrect answers, versus 52.7% with 40 as deployed (the guardrails cost 11 correct answers and prevented 5 incorrect ones).

## Generation health

baseline: 150 LLM calls, 11 stopped at the token limit, 16 re-sampled after a runaway generation, 0 old tool results shortened to fit the context window.
agent: 1218 LLM calls, 0 stopped at the token limit, 0 re-sampled after a runaway generation, 35 old tool results shortened to fit the context window.

## Judge validation

Blind manual grading of 40 judged answers (stratified by system and question type): the LLM judge agrees on 82% of labels (Cohen's kappa 0.72) and on 90% of correct-vs-not decisions (kappa 0.80). Judge said correct where the manual grade did not: 0; the reverse: 4. Accuracy on this sample: manual 57%, judge 48%.
