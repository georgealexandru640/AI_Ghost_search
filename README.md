# Ghost V1

Ghost V1 is a compact local research prototype built around seven small Transformer language models that collaborate on Python error diagnosis and repair.

The system contains a shared TinyGhost architecture and separate checkpoints for:

- base language modeling;
- diagnosis;
- two repair strategies;
- test proposal;
- critique;
- coordination.

## Architecture

```text
broken code + error log
          |
      diagnostic
       /      \
 repair A    repair B
       \      /
      ghost test
          |
        critic
          |
      coordinator
          |
 safety checks + human confirmation
```

The models do not modify the original source file directly. A proposed patch must pass structural validation, Python syntax checks, QA and security gates, and explicit human confirmation.

## Repository layout

```text
ghost_v1/
  tinyghost.py    BPE tokenizer, Transformer and text generation
  pipeline.py     data loading, training, messaging and command-line interface
  diagnostic.py   diagnosis role contract
  repair_a.py     first repair strategy
  repair_b.py     second repair strategy
  ghost_test.py   test proposal role
  critic.py       risk and quality review role
  coordinator.py  final candidate selection role
  project.json    configurations, tokenizer, datasets and evaluation metadata
  models.pt       seven local checkpoints
  tests.py        automated tests
```

## Requirements

- Python 3.11 or newer
- PyTorch 2.6 or newer
- Git LFS for downloading `models.pt`

## Setup

```powershell
git lfs install
git clone <repository-url>
cd ghost-v1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Usage

Inspect the registered models:

```powershell
python -m ghost_v1.pipeline info
```

Validate the project data and checkpoints:

```powershell
python -m ghost_v1.pipeline validate
```

Run the deterministic pipeline demonstration:

```powershell
python -m ghost_v1.pipeline demo --case case-001
```

Run the tests:

```powershell
python -m pytest .\ghost_v1\tests.py -q
```

## Scope and limitations

Ghost V1 is an educational research prototype trained on a small synthetic dataset and designed to run locally on CPU. It demonstrates the complete architecture, model specialization, checkpoint handling, structured role contracts and guarded collaboration pipeline. It is not a production code-repair system, and its holdout results should not be compared with large commercial language models.


