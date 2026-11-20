# SensorFM — agent instructions

Standalone project (not part of resolute or trialspec). Full plan: plan.md (read §0 and your task).
Free-tier only: $0 ever. GPU work runs on Kaggle Notebooks (see docs/decisions/000-free-tier-compute.md).
Never propose paid compute or paid APIs.

Commands: make setup | make lint | make test | make pretrain-debug | make data-debug
Layout: src/sensorfm/{utils,data,downstream,model,pretrain,kernels,finetune,baselines,eval,export,serving,monitoring}
Kaggle notebooks/scripts: kaggle/

Rules:
- Never fabricate results. Numbers flow scripts → results/*.json → generated reports. Targets only in docs/targets.md.
- Verify external facts (dataset locations, library APIs) at runtime; record in docs/decisions/.
- Every training path must run on CPU with the tiny config + synthetic data in < 2 min.
- Precision is configurable (fp32 | fp16 | bf16): Kaggle T4/P100 lack bf16.
- Local disk is tiny: never download corpora locally; keep data/ for debug artifacts only.
- Python 3.11, type hints, ruff + mypy --strict on src/, pytest.
- Stop at every 🛑 HUMAN CHECKPOINT in plan.md.
- Finish a task = tests + lint pass → append one line to PROGRESS.md → commit "phaseN.taskM: <summary>".
