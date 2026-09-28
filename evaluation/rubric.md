# Marking rubric for generated content

Written before any test-set output was generated. Items are marked against the sources the system actually received (notes, Whisper transcript, OCR slide text), which are shown next to each item on the rating sheet. Items are presented under random IDs, without the generation mode.

## 1. Faithfulness (every key concept, flashcard and quiz question)

Based on the intrinsic/extrinsic hallucination distinction in Maynez et al. (2020) and Ji et al. (2023).

| Label | Meaning |
|---|---|
| **S**: supported | Every claim is stated in the sources or follows directly from them. Paraphrase is fine. |
| **P**: partly supported | The main claim is supported, but a detail is added from outside the sources or is slightly distorted. |
| **U**: unsupported | The main claim is not in the sources (extrinsic) or contradicts them (intrinsic). |

For a quiz question, the question, the marked answer, every correct option and the explanation are all checked. A multiple-choice distractor is allowed to be false.

## 2. Quiz question quality (each quiz question)

Criteria adapted from the linguistic, educational and psychometric categories in Kurdi et al. (2020) and the item-writing flaws reported by Camarata et al. (2025).

| Criterion | 0 | 1 | 2 |
|---|---|---|---|
| **Answerable** from the sources | No | Only partly, or needs outside knowledge | Yes |
| **Correct** answer | Wrong | Ambiguous (e.g. more than one option is defensible) | Correct and unambiguous |
| **Key point** tested | Trivial or irrelevant | A minor detail | One of the bundle's key points or an important concept |

**Flaws** (recorded separately; more than one allowed):
- `multiple-correct`: more than one option is correct according to the sources
- `giveaway`: the question gives away its own answer
- `tautology`: the answer only restates the question
- `vague`: the question has no single clear answer

## 3. Flashcards (usefulness flag, in addition to faithfulness)

- `useful`: the back gives information that the front asks for.
- `tautology`: the back only restates the front (e.g. "What is the control unit? Control Unit.").
- `trivial`: a correct card about something not worth revising.

## 4. Summary coverage (each summary)

Each summary is checked against the bundle's 5 key points, which were written before generation. A point scores 1 if it is clearly stated, 0.5 if only partly or vaguely, and 0 if absent. Coverage = total / 5.

## 5. Second rater

A second adult rater re-marks a random 20% of items after a short calibration on 5 practice items (Fabbri et al., 2021). Agreement is reported as percentage agreement and Cohen's kappa (Cohen, 1960; McHugh, 2012), with a target of at least 0.60. Percentage agreement is reported alongside kappa because, if most items are "S", kappa can be low even when agreement is high (Viera & Garrett, 2005).

Prepared (`ratings/second_rater.csv`) but not carried out; see report Section 5.6.3.

## Amendment made during marking

While marking bundle B2, a failure appeared that none of the three faithfulness labels describes: items whose explanation, answer or back contains only a placeholder copied from the source tag (for example `ATP synthase: [audio transcript]`). They make no claim, so they are not hallucinations, but they are unusable. A fourth label was added:

| Label | Meaning |
|---|---|
| **E**: empty | The item contains a placeholder or no content where the answer, explanation or back should be. |

The flashcard flag `empty` was added for the same reason. For a quiz question with an empty answer, *correct* is marked 0. When reporting faithfulness, E is counted separately and does not count as supported. Items already marked (bundle B1) contained no placeholders, so they were unaffected.
