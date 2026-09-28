# Heuristic evaluation of the RevisionFlow interface

**Method:** Nielsen's ten usability heuristics, with his severity scale:

| Severity | Meaning |
|---|---|
| 0 | Not a problem |
| 1 | Cosmetic |
| 2 | Minor |
| 3 | Major |
| 4 | Catastrophe |

The walkthrough used the example material, covering every path: processing, editing, generating, the quiz, flashcards, sessions and errors. Errors were checked in two ways:
- In the browser, by loading a file from another app.
- Through the unit tests, which simulate Ollama being unavailable and the model returning invalid replies. The warning text shown to the student is the one produced by those code paths.

**Evaluator:** myself. Because I built the interface, this review is biased towards it. It was used to find problems to fix, and since the user study was not run, it is also the only usability evidence.

| # | Heuristic | Finding | Severity | Action |
|---|---|---|---|---|
| 1 | Visibility of system status | Each stage has its own spinner naming the model (for example "Transcribing audio with Whisper (base.en)"). The first image run waits about 40 s while BLIP loads, and generation takes about 30 s, with no sign of how long is left. | 2 | Add the expected wait to the generation spinner. |
| 2 | Match with the real world | Step 2 labels use model names ("Whisper", "OCR", "BLIP") that students may not know. The settings expander uses internal terms ("schema", "two_step"). | 2 | Keep the model names, but add plain descriptions in the step 2 caption. The settings are advanced options and stay as they are. |
| 3 | User control and freedom | Intermediate text can be edited, and "Start a new session" resets everything. Clicking **Process material** a second time silently replaces any edits made in step 2. | 2 | Warn next to the button when edits would be replaced. |
| 4 | Consistency and standards | The three steps are numbered and styled the same way. The two download buttons sit in different places (flashcards in their tab, session at the bottom). | 1 | Leave as it is. Each sits next to the content it saves. |
| 5 | Error prevention | **Process material** is disabled until there is input. **Check answer** is disabled until an answer is given. File types are restricted. The topic warning appears before generation. There is no warning that very long recordings will take a long time. | 2 | Mention in the README (listed under Limitations). The example material uses short clips. |
| 6 | Recognition rather than recall | The tabs, the step headings and the source tag on every item keep what matters visible. | 0 | - |
| 7 | Flexibility and efficiency | Only one audio file and one image can be added per session, but a real lecture has many slides. | 2 | A known scope limit, reported as further work. |
| 8 | Aesthetic and minimalist design | Step 2 shows up to four large text boxes, so the page gets long. The details tab keeps the technical tables out of the way. | 1 | Leave as it is. |
| 9 | Recognise, diagnose and recover from errors | If Ollama is not running, the warning reads "Could not reach Ollama (ConnectionError)". It is accurate but does not say how to fix it. Weak-support flags sit in collapsed expanders and are easy to miss. | 3 | Make the Ollama message say what to do. Keep the flag count visible at the top of step 3 (already shown). |
| 10 | Help and documentation | There are tooltips on the example toggle and the generation mode, and the captions explain each step. There is no separate help page. | 1 | The README covers setup. The captions explain each step of the workflow. |

## Changes made after this review
- **Heuristic 9 (severity 3):** the fallback warning now tells the student to open the Ollama app or run `ollama serve`, then generate again.
- **Heuristic 3:** a caption under **Process material** warns that processing again replaces the edited text in step 2.
- **Heuristic 1:** the generation spinner now says that generation usually takes about 30 seconds.
- **Heuristic 2:** the step 2 caption explains in plain words what each box is.
