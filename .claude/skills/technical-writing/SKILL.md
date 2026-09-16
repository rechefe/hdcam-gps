---
name: technical-writing
description: House style for every word written in this repo - module docstrings, inline comments, test names, README sections, notebook markdown, commit messages. Use before writing or editing any prose here. Enforces one claim per sentence and deletes sentences that carry none.
---

# Technical writing

The reader is a signal processing engineer who has never seen this code, and is
skimming. Every sentence costs them a second and has to earn it.

## The test for a sentence

After reading it, what does the reader know that they did not know before?
"Nothing" means delete it. Apply this to every sentence, including the ones you
like.

Four ways a sentence fails:

- **Restatement.** It repeats the previous sentence with different words.
- **Announcement.** "This section explains how the codebook is built." The
  section is right there doing that.
- **Padding.** "It is important to note that", "in order to", "it should be
  mentioned that", "as we can see".
- **Hedge stack.** "may potentially", "could possibly", "generally tends to".
  Commit to the claim, or state the condition under which it holds.

## Rules

1. **One claim per sentence.** Two claims joined by "and" are two sentences.
2. **Lead with the claim.** Caveats and conditions come after it, not before.
3. **Name the thing.** "the vote reduction", not "this mechanism". "PRN 19", not
   "another satellite".
4. **Numbers beat adjectives.** "38 dB-Hz against 40" beats "somewhat more
   sensitive". "two orders of magnitude slower" beats "much slower".
5. **Give the reason once.** Why a design exists is the most valuable thing you
   can write and the easiest thing to repeat by accident.
6. **Say what is wrong, not only what is right.** A known limit, a wrong
   assumption or a measurement that disagrees with theory is worth more than
   another paragraph of description.
7. **Delete on sight:** leverage, utilize, robust, comprehensive, seamless,
   powerful, simply, basically, essentially, very, in order to.

## Where each kind of prose goes

| Where | What it says | Length |
|---|---|---|
| Module docstring | Why the module exists, and the one thing a reader would otherwise assume wrongly | 3-12 lines |
| Class / function docstring | One line of contract, then Google style `Args:` / `Returns:` | 1 line + args |
| Inline comment | Why this line, never what it does | 1 line |
| Test docstring | What varies, what is held fixed, what should happen | 1-3 lines |
| Notebook markdown | The single idea the next cell demonstrates | 2-6 lines |
| Commit subject | The change, in the imperative, under 72 characters | 1 line |

Docstrings state the contract. Module docstrings state the reasoning. Keep them
separate: a function docstring that argues is too long, a module docstring that
lists methods is redundant with the code.

## Worked example

Bad:

```python
"""This module provides functionality for calibrating the HdCam classifier.

It contains various utilities and helper functions that can be used in order to
determine optimal parameter values. The calibration process is important because
using incorrect parameters may potentially lead to suboptimal performance.
"""
```

Every sentence fails the test. The reader learns that a calibration module
calibrates.

Good — the real docstring of `calibrate.py`:

```python
"""Calibrating an HdCam classifier against an operating point.

The threshold and the vote rule of OneBitHdCamClassifier can be derived from a
chance model, and that model is wrong in a way that matters: it treats every non
matching row as a coin flip, when a strong satellite cross correlates with the
other PRNs' codes by a fixed amount that repeats on every look. The measured
false alarm rate is then orders of magnitude worse than the model predicts.

So this module measures instead of assuming...
"""
```

The reader now knows the module exists because a formula lied, and roughly by how
much. That is the thing they could not have guessed from the code.

## Before you finish

Reread what you wrote and delete a quarter of it. The sentences that survive are
the ones that had a point.
