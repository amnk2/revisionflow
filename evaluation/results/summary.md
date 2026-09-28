# Technical evaluation summary

## Speech-to-text (test set, 8 clips, 33-47 s each)

| model | mean_wer | max_wer | clean_wer | noisy_wer | mean_seconds | real_time_factor |
|---|---|---|---|---|---|---|
| base | 0.022 | 0.069 | 0.012 | 0.031 | 1.408 | 0.034 |
| base.en | 0.028 | 0.089 | 0.015 | 0.042 | 1.099 | 0.027 |
| small.en | 0.02 | 0.089 | 0.009 | 0.031 | 2.976 | 0.075 |

### Noise stress test: mean WER by signal-to-noise ratio

| snr_db | base | base.en | small.en |
|---|---|---|---|
| 20 | 0.033 | 0.031 | 0.016 |
| 10 | 0.037 | 0.038 | 0.022 |
| 5 | 0.065 | 0.056 | 0.02 |
| 0 | 0.118 | 0.101 | 0.054 |

Segments flagged as probably silent across all stress runs: 0

### WER at 0 dB by accent (mean of three models)

| accent | wer |
|---|---|
| British | 0.05 |
| American | 0.073 |
| Indian | 0.086 |
| Australian | 0.096 |
| Irish | 0.129 |
| South African | 0.168 |

## Image stage (test set, 8 slides)

| method | mean_key_term_recall | min_key_term_recall | mean_cer |
|---|---|---|---|
| blip_caption | 0.012 | 0.0 | nan |
| ocr+caption | 1.0 | 1.0 | nan |
| ocr_binarised=False | 1.0 | 1.0 | 0.034 |
| ocr_binarised=True | 1.0 | 1.0 | 0.035 |

### OCR degradation stress test: mean key-term recall (8 synthetic slides)

| degradation | without own binarisation | with own binarisation |
|---|---|---|
| original | 1.0 | 1.0 |
| scale 0.5 | 1.0 | 1.0 |
| scale 0.35 + blur | 0.862 | 0.638 |
| scale 0.25 + blur | 0.0 | 0.0 |

### Real diagram (preliminary-report screenshot), no binarisation

| degradation | width_px | key_term_recall | cer | mean_confidence |
|---|---|---|---|---|
| original | 1736 | 0.8 | 0.156 | 84.5 |
| scale 0.5 | 868 | 0.7 | 0.188 | 84.8 |
| scale 0.35 + blur | 607 | 0.1 | 0.476 | 94.4 |
| scale 0.25 + blur | 434 | 0.1 | 0.478 | 89.6 |

## Relevance check (8 matched, 8 mismatched bundles)

Lowest pairwise similarity per bundle:

| case | min embedding | min tfidf | mean embedding | mean tfidf | max embedding | max tfidf |
|---|---|---|---|---|---|---|
| matched | 0.34 | 0.009 | 0.505 | 0.074 | 0.613 | 0.176 |
| mismatched | 0.009 | 0.0 | 0.039 | 0.001 | 0.092 | 0.011 |

| method | threshold | accuracy | precision | recall | best accuracy at any threshold |
|---|---|---|---|---|---|
| embedding | 0.3 | 1.0 | 1.0 | 1.0 | 1.0 |
| tfidf | 0.3 | 0.5 | 0.5 | 1.0 | 0.875 |

### Same-subject mismatch (both computer science)

| bundle | audio_from | method | min_score | flagged_at_threshold |
|---|---|---|---|---|
| B1 | B8 | embedding | 0.227 | True |
| B1 | B8 | tfidf | 0.0 | True |
| B8 | B1 | embedding | 0.127 | True |
| B8 | B1 | tfidf | 0.0 | True |

## Generation (8 bundles x 2 modes x 3 runs)

| mode | runs | valid_first_try | fallback | mean_repairs | key_concepts | flashcards | quiz | quiz_mcq | generation_s | generation_sd | flagged_weak |
|---|---|---|---|---|---|---|---|---|---|---|---|
| schema | 24 | 1.0 | 0.0 | 0.0 | 6.08 | 7.58 | 5.75 | 3.0 | 23.91 | 3.65 | 0.75 |
| two_step | 24 | 1.0 | 0.0 | 0.0 | 6.62 | 6.75 | 4.75 | 2.5 | 36.31 | 7.97 | 0.38 |

Runs identical across all 3 repeats: 15 of 16 bundle/mode pairs (temperature 0, seed 42).

## End-to-end time per bundle (models loaded, default settings, 24 runs)

| stage | mean | std | max |
|---|---|---|---|
| speech | 1.1 | 0.15 | 1.44 |
| OCR | 0.27 | 0.06 | 0.36 |
| caption | 0.84 | 0.52 | 1.99 |
| relevance | 0.03 | 0.0 | 0.03 |
| generation | 23.91 | 3.65 | 32.79 |
| grounding | 0.05 | 0.01 | 0.06 |
| total | 26.2 | 3.45 | 34.51 |

## Faithfulness (share of items)

| mode | kind | S | P | U | E |
|---|---|---|---|---|---|
| schema | flashcard | 0.8 | 0.117 | 0.083 | 0.0 |
| schema | key concept | 0.75 | 0.146 | 0.104 | 0.0 |
| schema | quiz question | 0.848 | 0.109 | 0.043 | 0.0 |
| two_step | flashcard | 0.87 | 0.037 | 0.019 | 0.074 |
| two_step | key concept | 0.698 | 0.094 | 0.038 | 0.17 |
| two_step | quiz question | 0.763 | 0.053 | 0.0 | 0.184 |

Overall share:

| mode | S | P | U | E |
|---|---|---|---|---|
| schema | 0.799 | 0.123 | 0.078 | 0.0 |
| two_step | 0.779 | 0.062 | 0.021 | 0.138 |

Counts:

| mode | S | P | U | E | items |
|---|---|---|---|---|---|
| schema | 123 | 19 | 12 | 0 | 154 |
| two_step | 113 | 9 | 3 | 20 | 145 |

Items carrying a transcription error, extrinsic-but-correct additions, placeholders:

| mode | asr_error | extrinsic_correct | placeholder |
|---|---|---|---|
| schema | 7 | 7 | 0 |
| two_step | 5 | 0 | 20 |

Share supported by bundle:

| bundle | schema | two_step |
|---|---|---|
| B1 | 0.94 | 1.0 |
| B2 | 0.76 | 0.25 |
| B3 | 1.0 | 0.85 |
| B4 | 0.65 | 0.88 |
| B5 | 0.6 | 0.94 |
| B6 | 0.85 | 0.71 |
| B7 | 0.59 | 0.55 |
| B8 | 1.0 | 0.94 |

## Quiz question quality (0-2 per criterion)

| mode | answerable | correct | key_point | share_with_flaw |
|---|---|---|---|---|
| schema | 1.98 | 1.72 | 1.74 | 0.13 |
| two_step | 1.97 | 1.47 | 1.66 | 0.11 |

## Flashcard usefulness

| mode | empty | tautology | trivial | useful |
|---|---|---|---|---|
| schema | 0.0 | 0.0 | 0.033 | 0.967 |
| two_step | 0.037 | 0.037 | 0.0 | 0.926 |

## Summary key-point coverage

| mode | mean | min | max |
|---|---|---|---|
| schema | 0.4 | 0.2 | 0.7 |
| two_step | 0.5 | 0.3 | 0.7 |

## Grounding check against manual labels (299 items; 63 marked P, U or E)

| threshold | flagged | precision | recall |
|---|---|---|---|
| 0.3 | 1 | 1.0 | 0.016 |
| 0.35 | 1 | 1.0 | 0.016 |
| 0.4 | 2 | 0.5 | 0.016 |
| 0.45 | 4 | 0.25 | 0.016 |
| 0.5 | 9 | 0.444 | 0.063 |
| 0.55 | 19 | 0.474 | 0.143 |
| 0.6 | 43 | 0.442 | 0.302 |
| 0.65 | 75 | 0.387 | 0.46 |
| 0.7 | 111 | 0.405 | 0.714 |
| 0.75 | 159 | 0.308 | 0.778 |
| 0.8 | 192 | 0.266 | 0.81 |
| 0.85 | 235 | 0.251 | 0.937 |

Similarity score by manual label:

| faithfulness | count | mean | min | max |
|---|---|---|---|---|
| E | 20.0 | 0.618 | 0.225 | 0.957 |
| P | 28.0 | 0.709 | 0.452 | 0.94 |
| S | 236.0 | 0.759 | 0.394 | 0.981 |
| U | 15.0 | 0.625 | 0.492 | 0.829 |

