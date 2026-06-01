# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuMF (Neural Matrix Factorization) recommendation system trained on the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset (Electronics and Musical Instruments categories). The model combines Generalized Matrix Factorization (GMF) and a Multi-Layer Perceptron (MLP) into a unified NeuMF architecture.

## Environment Setup

**Preferred:** Nix flake (provides Python 3.12 + all dependencies):
```bash
nix develop        # or: direnv allow  (if direnv is hooked into your shell)
```

**Alternative:** venv + pip:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `HF_TOKEN` for authenticated HuggingFace downloads.

## Commands

| Task | Command |
|------|---------|
| Download raw data | `python main.py` |
| Run training pipeline | `python -m src.pipeline.pipeline` |
| Lint | `ruff check .` |
| Format | `black .` |
| Type-check | `mypy src/` |

## Architecture

```
main.py                    # One-shot data downloader (HuggingFace Hub → data/raw/explicit/)
src/
  data/
    load_data.py           # Load CSVs into DataFrames
    preprocess.py          # Split train/val/test, encode user/item IDs
  features/
    build_features.py      # Negative sampling, interaction matrix construction
  models/
    train.py               # NeuMF training loop (GMF + MLP branches)
    predict.py             # Inference / hit-rate evaluation
    save_load.py           # torch.save / torch.load helpers
  pipeline/
    pipeline.py            # End-to-end orchestration: load → preprocess → train → evaluate
models/                    # Saved model checkpoints (.pt files)
data/
  raw/explicit/            # Downloaded CSVs: electronics.csv, musical_instrument.csv
```

**Data flow:** `main.py` downloads JSONL from HuggingFace and writes `{category}.csv` with columns `[user_id, parent_asin, rating, timestamp]`. The pipeline then preprocesses these into integer-encoded user/item pairs, constructs negative samples, trains the NeuMF model, and evaluates with Hit Rate / NDCG metrics.

**Model architecture:** NeuMF fuses two parallel branches — GMF (element-wise product of embeddings) and MLP (concatenated embeddings through fully-connected layers) — whose outputs are concatenated before a final sigmoid output layer.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **train-model** (212 symbols, 256 relationships, 3 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/train-model/context` | Codebase overview, check index freshness |
| `gitnexus://repo/train-model/clusters` | All functional areas |
| `gitnexus://repo/train-model/processes` | All execution flows |
| `gitnexus://repo/train-model/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
