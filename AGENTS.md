# Train Tycho

Eval, training data and fine-tuning for **Tycho**, the small model inside [Hven](../hven): a private Android assistant
that runs entirely on-device and does things on the phone (calendar, contacts, messages, reminders, timers…).
The goal is a 2B model that picks the right tool with the right values, reliably, on a phone.

## Goals

- **Measure first.** The eval runs the exact model file the app runs (Q4_K_M GGUF), with the app's prompt, grammar
  and sampling, against a fake phone. A number that doesn't match the phone is worthless.
- **Code knows the right answer.** Each training case comes from a recipe: code draws the phone, the request and the
  correct outcome; the teacher model only writes the user's words and plays the assistant. Code checks every run;
  no LLM judge (a model judging itself shares its blind spots).
- **Eval cases never leak into training.**

## Contract with the app

These must stay identical to what `../hven` ships; a change on one side needs the same change on the other.

- `app/tools.json`: the tool definitions. This repo owns it; the app gets a copy.
- `app/hven.gbnf`: the tool-call grammar, generated from `tools.json` (`uv run python -m app.grammar`).
- `app/prompt.py`: the prompt format, twin of the app's `ModelConfig.kt` (system turn, date list, tool responses).
- `app/sim.py`: each fake tool behaves like its Android version.
- `llama-server` is built from the same llama.cpp as the app (`../hven/third_party/llama.cpp`, v0.7.0).

## Structure

- `app/`: the contract above, plus `grammar.py` (tools → `hven.gbnf`, and a regex twin of the grammar).
- `eval.py`, `cases/`: the eval runs the hand-written cases (`core.jsonl`) on the base phone (`phone.json`).
- `gen/`: what a generated case is made of. `ingredients.json` and `phones.py`: names, numbers, events and notes from
  many regions, random phones. `recipes.py`: recipe kinds by category; code computes each expected outcome.
  `writer.json`: message styles and the examples the teacher sees. `checks.py`: does a message say what the recipe needs.
- `generate.py`: pass 1, the teacher writes the user's messages; pass 2, it plays the assistant live on the fake
  phone, and `check()` keeps only correct runs. Each record keeps the teacher's tokens and time.
- `stats.py`: summary of a data folder and `cases.csv` for plots. `qwen35Btest.py`: the 300-case pilot of a teacher.
  `smoke.py`: environment and teacher checks when the repo lands on a new machine.
- `out/` (results, data, pilot) and `build/` (llama-server): generated, gitignored.

## Running

Python 3.14 via `uv`, standard library only.

- llama-server: `cmake -S ../hven/third_party/llama.cpp -B build/llama -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_USE_PREBUILT_UI=OFF && cmake --build build/llama --target llama-server -j`
- Eval: `build/llama/bin/llama-server -m MODEL.gguf -c 4096`, then `uv run eval.py [--think] [--runs N] [--only PREFIX]`.
- Data: a teacher behind an OpenAI-style `/v1/completions` server, then
  `uv run generate.py messages --count N` and `uv run generate.py runs --count N` (`--workers`, `--think-budget`, `--out`).
- Previews: `uv run python -m gen.recipes`, `uv run python -m gen.phones`.
- Stats: `uv run stats.py [DIR]` (default `out/data`).

## Working rules

- **KISS.** The simplest thing that works and can be explained simply.
- **Discuss before implementing.** The human proposes, the agent critiques and proposes alternatives, until the human
  fully understands the chosen approach. No code for a design that hasn't been agreed.
- **Peers, not order-taking.** The agent follows the human's final decisions, but says clearly when something is wrong, and why.
- **The human must understand all code.** Code nobody can explain doesn't get merged.
- **Commits:** the human makes most commits. Before committing, the agent always asks; if the human says yes, the
  commit gets the agent's `Co-Authored-By` trailer, otherwise it carries no agent attribution.
- **Design notes and progress** live outside the repo, in `../documentation/` (`AGENT-design.md`, `TODO.md`).
  Keep `TODO.md` current.

## Code rules

- Pin versions (`pyproject.toml`, `uv.lock`). Standard library unless a dependency clearly pays for itself.
- Comments, for new and changed code: concise and simple. File header at the top, at most 4 lines; one or two lines
  above each function; inside functions only a rare short one-liner. Short format examples are fine.
