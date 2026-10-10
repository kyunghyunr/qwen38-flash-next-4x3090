# 125B vs 27B, same family, program-graded

I ran this to decide whether the 125B on this box earns its keep over the 27B I was already using.
The harness has 21 tasks across 6 categories; every task is graded by running something, not by
reading the answer. Code tasks are executed and graded by their tests. The grader itself was checked
first: a reference solution scores full marks, a deliberately wrong one loses them.

19 of the 21 tasks ran. I stopped the run during the second repetition, so 97 records exist. Totals
differ between the two models because they completed different numbers of tasks before the stop, not
because anything failed. Repetitions are 1-2, so treat variance as unmeasured.

## Summary by setting

| setting | 125B | 27B | note |
|---|---|---|---|
| effort=medium | 67/67 (100%) · 12,469 tok | 56/56 (100%) · 14,981 tok | both perfect |
| thinking off | 20/20 (100%) · 1,114 tok · 6 s | 20/20 (100%) · 1,273 tok · 5 s | fastest |
| effort=xhigh | 58/58 (100%) · 25,735 tok · 180 s | 48/48 (100%) · 35,522 tok · 245 s | same accuracy, 1.4-2x the tokens |
| xhigh, 3 hard tasks, 20K budget | 1/3 tasks solved | 1/3 tasks solved | see below |

## Per task, effort=medium

| task | 125B | 27B | 125B tok / time | 27B tok / time |
|---|---|---|---|---|
| code_lru (O(1) LRU) | 8/8 | 8/8 | 554 / 3.1 s | 474 / 2.1 s |
| code_bugfix (inclusive-range bug) | 5/5 | 5/5 | 295 / 1.7 s | 378 / 1.8 s |
| code_sql (top customer by month) | 1/1 | 1/1 | 880 / 4.9 s | 773 / 4.0 s |
| code_parse (recursive-descent parser) | 11/11 | 11/11 | 627 / 3.3 s | 1,633 / 7.8 s |
| code_dp (edit distance + script) | 6/6 | 6/6 | 1,768 / 9.5 s | 3,728 / 19.3 s |
| code_async (token bucket) | 3/3 | 3/3 | 861 / 5.2 s | 1,119 / 6.9 s |
| code_regex | 0/0 (budget exhausted) | 2/2 | 1,591 / 8.3 s | 1,295 / 5.7 s |
| code_csv (RFC4180) | not run | not run | | |
| math_bayes / logic_seating | 1/1, 5/5 | 1/1, 5/5 | | |
| korean_jamo (ㄹ final) | 1/1 | 1/1 (one rep answered 11) | 987 / 6.1 s | 1,115 / 7.2 s |
| korean_jamo2 (ㄴ final) | 1/1 | 1/1 | 470 / 3.2 s | 928 / 5.3 s |
| korean_particle (은/는) | 12/12 | 12/12 | 490 / 3.4 s | 456 / 3.9 s |
| json_schema / format / instr_multi | all pass | all pass | | |
| tool_call | 4/4 | 4/4 | 134 / 0.8 s | 172 / 1.2 s |
| needle_8k (3 needles) | 3/3 | 3/3 | 242 / 2.1 s | 371 / 5.4 s |

## What I took from it

1. **No coding gap at this difficulty.** LRU, bug fixing, SQL, a recursive-descent parser, an edit
   distance script, an async rate limiter: both models scored full marks. Any belief that the 125B
   is the better coder was not reproduced here. The task set is too easy to show a difference.
2. **xhigh is a failure mode, not an upgrade.** Given 20K tokens of budget on the three hardest
   tasks, the 125B emitted wrong code (`UnboundLocalError`) on `code_parse` and the 27B emitted no
   code at all; on `code_dp` neither produced code, one degenerating into `0000...` and the other
   repeating "Potential issue:". The same tasks passed 11/11 and 6/6 at medium, and passed with
   thinking off.
3. **medium is the sweet spot**: full accuracy at 3-19 s per task. Thinking off keeps the accuracy
   and lands at 0.8-2.6 s.
4. **The 27B spends 1.4-2.1x the tokens and 1.3-2.0x the time** for the same answers. Its reasoning
   is longer and more consistent; the 125B's is shorter and more variable.
5. **The only visible difference was Korean final-consonant questions**, where the 125B got 2/2 and
   the 27B 1/2. Two samples is not evidence, and I am not going to pretend it is.
6. **Budget exhaustion is stochastic, not a model trait.** On `code_regex` at xhigh the 125B passed
   and the 27B ran out of budget; at medium it flipped. That is sampling noise, not a capability.

## Limits

- 1-2 repetitions, so no variance estimate. The budget-exhaustion and degeneration cases are
  probabilistic and I only saw each once or twice.
- Most tasks are at or below both models' ceiling, so the comparison is mostly a floor test.
  Harder tasks (multi-file refactors, hidden edge cases) are needed to separate them.
- Two tasks never ran, and long agentic work (many tool calls, growing context) was not tested at
  all. That is the case I actually care about and it is unmeasured here.

## What I use each for now

| work | pick | reason |
|---|---|---|
| general coding | either, effort=medium | hard tasks included, xhigh fails |
| long or complex code | either, budget >= 12K, medium | xhigh dropped the code entirely |
| fast turns, subagents | 27B | 0.8-7 s, half the tokens |
| design, options, discussion | 125B | longer answers, offers alternatives |
| short consistent output | 27B | stable length, less drift |
| tool calls, JSON, instruction following | either | all pass |
| Korean orthography | 125B (weak evidence) | 2/2 vs 1/2 |
| long-context recall | either | 3/3 both |
