# FinanceBench evaluation (150 questions)

Evidence on gold page: DocAgent = its cited evidence; baseline = its 5 retrieved passages.

| system | subset | n | accuracy | incorrect | refusal | acc. when answered | faithfulness | evidence on gold page | $/question | s/question | tool calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | all | 150 | **4.0%** | 15 | 10 | 28.6% | 100.0% | 41.3% | $0.000 | 148 | 0.0 |
| baseline | test split | 75 | **4.0%** | 6 | 6 | 33.3% | 100.0% | 42.7% | $0.000 | 146 | 0.0 |
| baseline | excluding pilot | 140 | **4.3%** | 14 | 9 | 30.0% | 100.0% | 39.3% | $0.000 | 147 | 0.0 |
| baseline | metrics-generated | 50 | **4.0%** | 7 | 4 | 22.2% | 100.0% | 38.0% | $0.000 | 148 | 0.0 |
| baseline | domain-relevant | 50 | **6.0%** | 5 | 3 | 37.5% | 100.0% | 38.0% | $0.000 | 146 | 0.0 |
| baseline | novel-generated | 50 | **2.0%** | 3 | 3 | 25.0% | 100.0% | 48.0% | $0.000 | 150 | 0.0 |
| agent | all | 150 | **12.0%** | 14 | 3 | 56.2% | 94.4% | 66.9% | $0.000 | 193 | 5.4 |
| agent | test split | 75 | **9.3%** | 6 | 3 | 53.8% | 90.9% | 71.4% | $0.000 | 200 | 5.6 |
| agent | excluding pilot | 140 | **12.9%** | 12 | 3 | 60.0% | 94.2% | 67.2% | $0.000 | 192 | 5.4 |
| agent | metrics-generated | 50 | **18.0%** | 4 | 0 | 69.2% | 90.9% | 69.6% | $0.000 | 174 | 5.4 |
| agent | domain-relevant | 50 | **16.0%** | 6 | 1 | 57.1% | 95.5% | 69.2% | $0.000 | 221 | 6.4 |
| agent | novel-generated | 50 | **2.0%** | 4 | 2 | 20.0% | 100.0% | 61.9% | $0.000 | 184 | 4.5 |

## Paired comparison (same questions)

DocAgent correct where the baseline is not: 14; the reverse: 2. Exact McNemar test p = 0.0042. Accuracy difference +8.0% (95% bootstrap CI +3.3% to +13.3%).

## Guardrails

Answers withheld by the citation/number guardrails: 5. Graded anyway, 2 of them would not have been correct.
