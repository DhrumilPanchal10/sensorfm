.PHONY: setup lint test test-all data-debug data-pretrain data-downstream pretrain-debug pretrain \
        finetune eval profile scale export bench serve report

PY := uv run
TODO = @echo "not implemented yet ($(1)) — see plan.md" && exit 1

setup:
	uv sync --group dev

lint:
	$(PY) ruff check src tests scripts
	$(PY) ruff format --check src tests scripts
	$(PY) mypy

test:
	$(PY) pytest -q -m "not slow and not gpu"

test-all:
	$(PY) pytest -q -m "not gpu"

data-debug:
	$(PY) python scripts/make_debug_data.py

data-pretrain:          # runs on Kaggle, not locally (disk)
	$(call TODO,Task 1.3)
data-downstream:
	$(call TODO,Task 1.5)
pretrain-debug:
	$(call TODO,Task 2.3)
pretrain:
	$(call TODO,Task 2.3)
finetune:
	$(call TODO,Task 4.2)
eval:
	$(call TODO,Task 4.3)
profile:
	$(call TODO,Task 3.1)
scale:
	$(call TODO,Task 3.3)
export:
	$(call TODO,Task 5.1)
bench:
	$(call TODO,Task 5.2)
serve:
	$(call TODO,Task 5.3)
report:
	$(call TODO,Phase 6)
