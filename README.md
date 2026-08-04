# DTV-REA — the digital-twin requirements agent

This program interviews you about a digital twin you want to build, one question
at a time, and turns your answers into a proper requirements document. You
describe the machine and the problem in your own words; it asks follow-up
questions, writes each thing you need down in a standard form, and hands you a
document at the end. The point of it is a promise it can actually keep: **it
never makes up a number.** If a tolerance, an accuracy or a percentage appears
in the finished document, you said it, and the document quotes the exact
sentence you said it in. When the program does not have a number, it asks you
for one rather than filling in something plausible — and if the language model
behind it tries to invent one anyway, a checker catches it and throws it out
before it can reach the page.

---

## Contents

1. [Before you start](#1-before-you-start)
2. [Setup](#2-setup)
3. [Running the agent](#3-running-the-agent)
4. [Where your files appear](#4-where-your-files-appear)
5. [Troubleshooting](#5-troubleshooting)
6. [How it works](#6-how-it-works)
7. [Design decisions, and why](#7-design-decisions-and-why)
8. [Honest limitations](#8-honest-limitations)

---

## 1. Before you start

You need **Python, version 3.11 or newer**. That is the only thing you need to
install by hand.

Download it from **<https://www.python.org/downloads/>** and run the installer.

> **On Windows, this bit matters:** on the very first screen of the installer
> there is a checkbox near the bottom that says **"Add Python to PATH"**. Tick
> it before you click Install. If you miss it, Windows will not be able to find
> Python later and nothing below will work. If you have already installed
> Python without ticking it, just run the installer again and choose "Modify".

On macOS, run the installer and accept the defaults; there is no equivalent
checkbox to worry about.

**How to open a terminal**, which is the window you type these commands into:

- **macOS** — press `Cmd + Space`, type `Terminal`, press Enter.
- **Windows** — press the Start button, type `PowerShell`, press Enter.

**One thing to know about the commands below.** Each one is a line you type and
then press Enter. Nothing happens until you press Enter. If a command prints a
lot of text, that is normal — you only need to worry if it says `error`.

First, move the terminal into this project's folder. Replace the path with
wherever you actually saved it:

**macOS / Linux**

```bash
cd ~/Documents/DTVR
```

**Windows**

```powershell
cd $HOME\Documents\DTVR
```

Everything from here on assumes your terminal is in that folder.

---

## 2. Setup

Three steps. Do them once.

### Step 1 of 3 — Create a virtual environment

A virtual environment is a private folder for this project's add-ons, so that
installing them cannot disturb anything else on your computer. This creates one
called `.venv`, and then *activates* it, which tells the terminal to use it.

**macOS / Linux**

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

**Windows**

```powershell
py -3 -m venv .venv
```

```powershell
.venv\Scripts\Activate.ps1
```

You will know it worked because your prompt now starts with `(.venv)`.

> You have to run the *activate* line again each time you open a new terminal
> window. The first line, which creates the folder, is only ever needed once.
>
> If Windows refuses to run the activate line, see
> [Troubleshooting](#activation-is-blocked-on-windows-powershell) — it is a
> one-line fix.

### Step 2 of 3 — Install the dependencies

This downloads the libraries the agent is built on. It takes a minute or two and
prints a lot of progress text.

**macOS / Linux**

```bash
python -m pip install -r requirements.txt
```

**Windows**

```powershell
python -m pip install -r requirements.txt
```

Yes — that one is genuinely identical on both. Once the virtual environment is
active, `python` means the right Python on either operating system.

### Step 3 of 3 — Add your Groq API key

The agent uses a language model called Llama 3.3, hosted by a company called
Groq, to phrase its questions. That needs a key. Get one free at
<https://console.groq.com/keys>.

First copy the example file to a real one:

**macOS / Linux**

```bash
cp .env.example .env
```

**Windows**

```powershell
copy .env.example .env
```

Then open the new `.env` file in any text editor (TextEdit, Notepad, VS Code —
anything) and put your key after the `=` sign, so the line reads:

```
GROQ_API_KEY=gsk_your_own_key_goes_here
```

Save the file and close it. That is all of setup.

> **You can skip this step entirely if you only want to try the demo.** The
> demo, the tests and the evaluation all run offline with no key and no
> internet connection.
>
> Your `.env` file is deliberately excluded from version control, so your key
> never gets committed or shared.

### If you would rather not type all that

There are scripts that do all three steps for you. They are exactly equivalent —
same commands, nothing extra.

**macOS / Linux**

```bash
bash scripts/setup.sh
```

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

---

## 3. Running the agent

**Try the offline demo first.** It plays back a scripted interview about a 3D
printer so you can see exactly what the agent does, end to end. It needs no key
and no internet:

```bash
python -m dtv_rea.cli --stub
```

**To do a real interview**, where the agent asks *you* the questions:

```bash
python -m dtv_rea.cli
```

The agent asks one question at a time. Type your answer and press Enter. Answer
in ordinary sentences — full paragraphs are fine and usually better, because the
agent can pull several requirements out of one good answer.

**To stop early**, type `stop` at any question and press Enter. You will still
get everything captured up to that point, clearly marked as unfinished. (`quit`
and `exit` work too, as does Ctrl-D on macOS or Ctrl-Z then Enter on Windows —
but typing `stop` is the reliable way on every platform.)

**To pick a session back up** if you had to leave in the middle — the agent
remembers where it was:

```bash
python -m dtv_rea.cli --session my-interview --resume
```

Give the session a name of your own so you can find it again:

```bash
python -m dtv_rea.cli --session my-interview
```

Other useful commands, all identical on both operating systems:

| What you want | Command |
|---|---|
| Run the test suite | `python -m pytest` |
| See all the scripted example interviews | `python -m dtv_rea.cli --list-personas` |
| Score the agent against the published case study | `python -m eval.run_eval --stub` |
| See every option | `python -m dtv_rea.cli --help` |
| Run the tests, demo and scoring in one go | `bash scripts/demo.sh` / `powershell -ExecutionPolicy Bypass -File scripts\demo.ps1` |

`python -m pytest` and `python -m eval.run_eval --stub` both work with **no API
key and no internet**. That is on purpose: the part of the agent that makes the
decisions is deliberately separate from the language model, so it can be tested
without one.

---

## 4. Where your files appear

Everything a session produces goes in one folder:

```
runs/<session_id>/
```

`<session_id>` is whatever you passed to `--session`, or a timestamp like
`session-20260802-143000` if you did not pass one. The demo above writes to
`runs/demo/`.

Inside that folder:

| File | What it is |
|---|---|
| **`requirements.md`** | **Start here.** The readable document — the requirements grouped by topic, each design goal printed with a quote of the exact sentence you said it in, and anything unresolved clearly marked. |
| `session.json` | The complete record of the session, for tooling. If the interview was stopped early, the very first line says `"status": "partial"`. |
| `snapshots.jsonl` | One line per turn showing how the document grew. This is what the evaluation reads. |
| `llm_calls.jsonl` | One line per call to the language model, for checking speed and cost. It records a *fingerprint* of each prompt, never your transcript, and never your key. |

There is also a `runs/checkpoints.db` file. That is how `--resume` works; you can
ignore it.

---

## 5. Troubleshooting

### "python not found" / "python is not recognized"

Almost always one of two things.

**You have not activated the virtual environment.** Look at your prompt: does it
start with `(.venv)`? If not, run the activate line from
[Step 1](#step-1-of-3--create-a-virtual-environment) again — you need it in
every new terminal window.

**Python is not installed, or Windows cannot find it.** Check with:

**macOS / Linux**

```bash
python3 --version
```

**Windows**

```powershell
py -3 --version
```

If that prints something like `Python 3.12.4`, Python is fine and the problem
was the virtual environment. If it says "not found", reinstall from
<https://www.python.org/downloads/> — and **on Windows, tick "Add Python to
PATH"** this time. Close and reopen your terminal afterwards; it will not pick
up the change until you do.

### "pip not found"

Use `python -m pip` rather than plain `pip` — that is why every command in this
README is written that way. `pip` on its own relies on a shortcut that is not
always set up, but `python -m pip` always works if Python does.

If `python -m pip` still fails, pip is missing from your install:

**macOS / Linux**

```bash
python3 -m ensurepip --upgrade
```

**Windows**

```powershell
py -3 -m ensurepip --upgrade
```

### Activation is blocked on Windows PowerShell

If `.venv\Scripts\Activate.ps1` gives you a red error mentioning **"running
scripts is disabled on this system"**, that is a Windows security default, not a
problem with this project. Fix it once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Answer `Y` when it asks. This lets you run scripts you wrote or downloaded
yourself, and leaves the stricter rules for everything else in place. You will
not have to do it again. Now run the activate line again.

If your workplace forbids changing that setting, you can skip activation
entirely and call the environment's Python directly:

```powershell
.venv\Scripts\python -m dtv_rea.cli --stub
```

### "GROQ_API_KEY is not set"

The agent could not find your key. Work through these:

1. **Is there a file called `.env` in the project folder?** Not `.env.example` —
   that is the template. You need a copy of it named exactly `.env`. Go back to
   [Step 3](#step-3-of-3--add-your-groq-api-key).

2. **Is the file actually called `.env`?** Windows sometimes saves it as
   `.env.txt` without telling you. In File Explorer turn on
   *View → File name extensions* and check. Rename it if it is wrong.

3. **Is the key on the line?** Open `.env`. The line should read
   `GROQ_API_KEY=gsk_...` with your key straight after the `=`. No spaces around
   the `=`, no quotation marks around the key.

4. **Are you in the right folder?** The agent looks for `.env` in the project
   folder. If your terminal is somewhere else, `cd` back
   ([see above](#1-before-you-start)).

5. **Is the key still valid?** Check at <https://console.groq.com/keys>.

Remember you do not need a key at all for `--stub`, `python -m pytest`, or
`python -m eval.run_eval --stub`. If you are just trying it out, use those.

### "You have used up your quota" / rate limit reached

A free Groq account has a **daily token allowance of 100,000**. One full
interview uses roughly **35,000 tokens**, so a free account is good for about
**two or three live interviews a day**. When you hit the limit the agent tells
you how long to wait.

Nothing is lost when this happens — every answer is already saved. Wait, then
pick up exactly where you stopped:

```bash
python -m dtv_rea.cli --session my-interview --resume
```

If you need more, raise the allowance at
<https://console.groq.com/settings/billing>. The offline demo, the tests and the
evaluation are unaffected — they never call Groq at all.

### Something else went wrong

First, ask the agent to check its own setup. This prints where it is looking for
everything and whether it found your key — it never prints the key itself:

```bash
python -m dtv_rea.cli --doctor
```

Then run the tests — they take a couple of seconds and need no key:

```bash
python -m pytest
```

If those pass, your installation is sound and the problem is with the interview
itself. If they fail, the output will name the problem; the most common cause is
that Step 2 did not finish, so try it again.

---

## 6. How it works

The governing rule of the whole design is:

> **Decisions in code. Language in the model.**

The language model does exactly two jobs: it phrases questions, and it puts your
answers into a structured form. Every *decision* — what to ask next, whether an
answer is acceptable, whether the interview is finished — is ordinary Python
that you can read, test and step through. That is why the whole decision-making
core runs and is tested with no model at all.

The interview is a loop:

```
ask what the twin is for
    -> propose how capable it needs to be, and check that with you   [you decide]
    -> loop:
         what topic is still uncovered?            (code decides)
         phrase a question about it                (model)
         you answer                                [you decide]
         pull requirements out of the answer       (model)
         check them                                (code: five checks)
         commit the good ones, flag the doubtful   (code)
    -> put every flag to you before finishing      [you decide]
    -> write the document
```

**The five checks** are the part that makes the promise keepable:

| Check | What it catches | What happens |
|---|---|---|
| **V1 Fabrication** | A number that is not in the sentence it claims to come from — or that cites the *agent* talking rather than you | Thrown out. The agent then asks you for the value. It never asks the model again: a regenerated number is still an invented number. |
| **V2 Vague predicate** | "The system shall minimize data loss" — no test could ever pass or fail that | Kept, but flagged for you to put a number on |
| **V3 Duplicate obligation** | Two requirements carrying the same obligation | Kept, but flagged. Only you can merge them; the agent never will |
| **V4 Orphan goal** | A number attached to a requirement that does not exist | Thrown out |
| **V5 Contradiction** | A requirement in a topic you already ruled out | Kept, but flagged, showing you both of your statements |

**You are asked at three points**, and there is no path around any of them:
confirming how capable the twin needs to be (with a plain-language description,
never just a label), every single interview answer, and every flag before the
document is finished. An agent that could finish a requirements document without
talking to a human would, by this project's definition, have made it up.

The reference case throughout is a published FDM 3D-printer digital twin
(Bitencourt et al. 2025, *International Journal of Production Research*,
DOI [10.1080/00207543.2025.2524516](https://doi.org/10.1080/00207543.2025.2524516)),
with its twelve requirements and six design goals. Two genuine defects in that
published set — one unverifiable "minimize", and one obligation stated twice —
are used as permanent tests that the checks still work.

---

## 7. Design decisions, and why

**LangGraph, not LangChain.** This agent is a state machine with cycles, decision
points in code, and pauses for a human. That is precisely LangGraph's shape:
cycles are its core primitive, state is typed and persistent, branching is plain
Python functions, and pausing and resuming durably is built in. Chains are
linear, keep their state implicitly in message history, and have no native
pause. The cost is a dependency and a learning curve over a hand-written loop;
what it buys is durable checkpointing, real interrupts, and a structure that
maps one-to-one onto the documented design.

**No RAG. The knowledge base is stuffed into the prompt.** The background
knowledge here is small and fixed — the four maturity levels, the four
requirement topics, a summary of the framework, and one worked example. Around
ten pages, against a 128,000-token context window. Retrieval would add a *silent*
failure mode: if the maturity definitions failed to retrieve on the exact turn
the agent proposes a level, the proposal would be unguided and the output would
still look completely normal. For a corpus this size that is risk with no
benefit.
**Revisit trigger:** adopt retrieval only if this knowledge base grows past the
context window — for instance if many more worked cases or full standards
documents are added.

**A hybrid dataset.** One real, citable ground-truth case (the published FDM
twin) anchors the accuracy measurements to something outside this project. Five
synthetic personas cover the paths the real case cannot reach: a stakeholder who
will not engage, one who does not know a number, one who contradicts himself,
one who never gives a number at all, and one who walks out. Synthetic is the
right call here because no public corpus of digital-twin requirements interviews
exists. Adding another real case needs no code — drop a `<case>.json` into
`data/ground_truth/` and a persona beside it.

**Human in the loop is the product, not a feature.** The stakeholder is the only
legitimate source of purposes, tolerances and authorisations. In the reference
case every single design goal was stakeholder-authorised and none was derivable
from anything else. So the three checkpoints are mandatory and unskippable.

**The research claim is enforced by the type system, not by asking nicely.** A
design goal's origin can only be recorded as `"stated"`. There is no
`"agent_derived"` option to select — the program would refuse to construct such
a requirement. Prompts ask the model not to invent numbers, but prompts are
requests; this is a wall.

Every knob is in one file, `dtv_rea/settings.py`: the model name, the
temperatures, the attempt cap (3), the duplicate threshold (0.6), and the
vague-word list.

---

## 8. Honest limitations

These are real, and they belong in any writeup of this work.

- **n = 1 real scenario.** There is exactly one reverse-engineered ground-truth
  case. That is not a sample, and no statistical claim should be made from it.
  Two or three more reverse-engineered cases are the planned fix.

- **V2 is a keyword heuristic.** It spots vague predicates from a list of words
  like "minimize" and "reliable". It has no idea whether a requirement is
  unverifiable for any other reason, and it is a first pass, not a judgement.

- **V3's threshold is tuned to the known case.** It sits at 0.6 because the
  published UR1.2 / UR1.7 pair scores exactly 0.60. It will sometimes flag pairs
  a human considers perfectly distinct. That is why V3 only ever raises a
  question and never merges anything by itself.

  Two guards keep that from becoming noise. A pair must share at least **three**
  content words, not just clear the ratio — otherwise two short requirements
  sharing only generic words ("collect", "job") score 0.67 and look like a
  duplicate when they are simply both short. And each requirement raises **at
  most one** question, about its closest match, so N similar requirements cost N
  questions rather than every one of their pairings.

- **Single stakeholder.** The agent interviews one person. It has nothing to say
  about reconciling several people who disagree with each other.

- **The demo and the tests measure the core, not the model.** A `--stub` run
  proves the routing, the checks, the edge-case rules and the output are
  correct. It proves nothing whatsoever about how Llama 3.3 actually behaves.
  Only a live run measures that.

- **Requirement wording is not scored automatically.** The evaluation matches
  requirement text literally. Judging whether a requirement *means* the same as
  the published one, and whether it is well written, needs a human rubric
  (adapted from Ronanki et al. 2023).

- **Scope.** This implements steps 1 and 2 of the DTV framework only — problem
  definition and requirements. Alignment with IEEE 29148 is deliberately out of
  scope.

---

## Project layout

```
dtv_rea/          the agent
  settings.py     every configuration knob, in one place
  state.py        the typed session state
  validator.py    the five checks - the research claim lives here
  prompts.py      the six prompts and the embedded knowledge base
  graph.py        the state machine
  llm.py          the model interface, and the offline scripted model
  groq_model.py   Llama 3.3 on Groq
  persona.py      scripted stakeholders
  runner.py       drives the interview from one pause to the next
  outputs.py      writes the document
  cli.py          the command line
data/
  ground_truth/   published cases to measure against
  personas/       scripted stakeholders
eval/             the evaluation harness and its report
tests/            the test suite - runs offline, no key
scripts/          optional setup and demo helpers, one pair per platform
```

Requires Python 3.11 or newer. Tested on macOS and Windows.
