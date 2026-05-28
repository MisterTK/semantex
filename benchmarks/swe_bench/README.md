# SWE-bench Verified Harness

Reproducible harness for benchmarking semantex against SWE-bench Verified.

## Setup
```
cd benchmarks/swe_bench
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional (recommended for production runs): set `HF_TOKEN` for Hugging Face
rate-limit headroom when loading `princeton-nlp/SWE-bench_Verified`.

Docker is required for the swebench evaluation harness (`swebench`).

## Run Phase A (100 instances, ~$1.5-2k)
```
python -m scripts.pre_index --phase a
python -m scripts.run --phase a --replicates 2
python -m scripts.submit --run-id <id>
```

## Run Phase B (500 instances, ~$9-13k)
```
python -m scripts.pre_index --phase b
python -m scripts.run --phase b --replicates 3
python -m scripts.submit --run-id <id>
```
