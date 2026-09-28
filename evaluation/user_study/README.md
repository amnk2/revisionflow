# User study: how to run it

**Status:** prepared but not run. The report explains in Section 5.4 why objective O7 is therefore not assessed.

**Aim:** measure whether RevisionFlow is usable and whether its transparency features (editable intermediate outputs, topic warning, weak-support flags) help students check the AI's work. This covers objective O7.

**Participants:** 8 adults (minimum 5), recruited from course peers, the module forum or friends who study at university level. No children, no vulnerable adults, and no one under the researcher's authority.

**Material:** the example material built into the app (sidebar toggle "Use the example material"). Participants never upload their own files, and no personal data other than the questionnaire answers is collected.

**Time:** about 20 minutes per person, in person or over a screen share.

## Before each session
1. Start Ollama and run `bash run.sh`. Open http://localhost:8501 and click **Start a new session**.
2. Give the participant `information_sheet.md` and `consent_form.md`. Continue only once consent is given (a form tick or signature).
3. Give them an anonymous ID (P1, P2, ...). Their name is never written next to their answers.

## During the session
Follow `facilitator_script.md`. For each task, record in `responses.csv`:
- whether it was completed: **1** = without help, **0.5** = with a hint, **0** = not completed
- the time taken in seconds (a phone stopwatch is fine)

Do not help unless the participant is stuck for more than a minute, and note any hint given. Also write down anything they say out loud that shows confusion or a useful reaction.

## After the session
1. The participant fills in the questionnaire (`questionnaire.md`, set up as a Google Form). Copy their answers into the same row of `responses.csv`.
2. Thank them and remind them they can withdraw until the report is submitted, using their ID.

## Analysis
```bash
venv/bin/python evaluation/user_study/analyse_user_study.py
```
This writes:
- `evaluation/results/user_study_summary.md`: SUS mean, standard deviation and 95% confidence interval; task completion; Likert medians and distributions
- `evaluation/figures/sus_scores.png`

The open comments are grouped into themes by hand, following Braun and Clarke's (2006) phases in a light form, as stated in the report.
