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
	test -f data/shards/debug/manifest.json || \
	$(PY) python scripts/build_corpus.py --synthetic-only --synthetic-series 512 \
		--synthetic-length 4096 --shard-bytes 2000000 --val-frac 0.1 --out data/shards/debug

data-pretrain:          # run on Kaggle (see kaggle/README.md), not locally: disk
	$(PY) python scripts/build_corpus.py --selection configs/data/lotsa_selection.yaml \
		--reference $${REFERENCE_DIR:-data/downstream/reference} --out $${SHARD_OUT:-data/shards/main}
data-downstream:        # ~0.6 GB; on Kaggle use DOWNSTREAM_ROOT=/kaggle/working/downstream
	$(PY) python scripts/prepare_downstream.py --root $${DOWNSTREAM_ROOT:-data/downstream}
pretrain-debug: data-debug   # tiny model, synthetic shards, CPU, < 2 min
	$(PY) python -m sensorfm.pretrain.train
pretrain:
	$(call TODO,Task 2.3)
finetune:
	$(call TODO,Task 4.2)
eval:
	$(call TODO,Task 4.3)
profile:                # ladder + traces (GPU: kaggle/run_ladder.sh); CPU smoke here
	$(PY) python -m sensorfm.pretrain.bench --steps 40 --warmup 5 --micro-batch 16 --profile
scale:                  # GPU only: kaggle/run_scaling.sh
	@echo "run kaggle/run_scaling.sh on a Kaggle T4 x2 session" && exit 1
export:                 # TAG=s20m CKPT=runs/.../step_N (or REGRESSOR=exports/s20m/rul.pt)
	$(PY) python scripts/deploy.py --tag $${TAG:-s20m} $${CKPT:+--ckpt $$CKPT} \
		$${REGRESSOR:+--regressor $$REGRESSOR}
bench:                  # re-time the exported ONNX variants of TAG on this machine
	$(PY) python bench/latency.py --task rul --out results/bench/latency_$${TAG:-s20m}.json \
		$(foreach v,fp32 int8_dyn int8_static int8_static_keep,--variant $(v)=exports/$${TAG:-s20m}/rul_$(v).onnx)
serve:                  # SENSORFM_MODEL=path/to/model.onnx SENSORFM_TASK=rul|forecast
	$(PY) uvicorn sensorfm.serving.app:app --port 8000
report:                 # regenerate README results block + docs/reports from results/
	$(PY) python scripts/build_report.py
