# Isolated rerun workflow

These reruns are separate from the historical V2 experiment. The timeout study compares
explicit **600- and 1,800-second per-HTTP-request timeout arms**; the historical timeout is
unknown. No result should be interpreted as reproducing the historical timeout.

## Safety and readiness

- All data are confined to `results/reruns/`; the runner refuses destinations outside this root.
- Schedules/configs are immutable once written. Resumption uses the saved schedule and skips
  only terminal run records. Interrupted attempts are journaled and retried at the same planned
  position with a new attempt number.
- A global OS lock serializes model-executing invocations of the new rerun runner. It does
  not control unrelated Ollama clients or historical runner processes; ensure those are not
  running.
- The runner does not download models. The current audit found `qwen2.5:7b` and `mistral:7b`;
  `llama3.1:8b` is absent (the installed `llama3:latest` is not an interchangeable tag).
- Before a pilot, check Windows memory with
  `Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory`,
  GPU adapters with `Get-CimInstance Win32_VideoController`, model tags with `ollama list`,
  and active model placement with `ollama ps`. `ollama ps` only reports a model that is already
  loaded; do not issue a prompt just to check placement. Inspect local Ollama server logs in
  `$env:LOCALAPPDATA\Ollama\server*.log` for backend/device details. A GPU adapter's presence
  alone does not establish that Ollama used it.
- This audit ran on Windows 11 Home with Python 3.11.0 and Ollama 0.40.6. It observed
  15.37 GiB physical RAM with 2.28 GiB free, an Intel Arc integrated GPU with about 2 GiB
  adapter RAM, and no active model in `ollama ps`. The local Ollama startup log explicitly
  dropped the integrated GPU and reported inference compute as CPU with 0 B VRAM. Thus GPU
  acceleration is not established and CPU inference is the current expectation. Recheck
  resources before any pilot; these values can change, and enabling a GPU backend would require
  a separate configuration decision.
- No live pilot or full run is authorized by these instructions.

## Prepare frozen schedules (offline; no inference)

```powershell
python src\runner_rerun.py --experiment latency --model qwen2_5_7b --prepare-only
python src\runner_rerun.py --experiment timeout --model qwen2_5_7b --prepare-only

python src\runner_rerun.py --experiment latency --model qwen2_5_7b --prepare-pilot-only
python src\runner_rerun.py --experiment timeout --model qwen2_5_7b --prepare-pilot-only
```

The latency full schedule has 750 jobs. The timeout full schedule has 1,500 paired jobs
(750 per arm). Each pilot schedule has 9 latency jobs or 18 timeout jobs (paired arms).
Schedules and frozen configs include SHA-256 fingerprints. Preparation refuses to overwrite
different frozen artifacts.

## Pilot (requires separate approval)

After confirming machine readiness and approving a live pilot:

```powershell
python src\runner_rerun.py --experiment latency --model qwen2_5_7b --pilot
python src\runner_rerun.py --experiment timeout --model qwen2_5_7b --pilot
```

The timeout command executes each paired 600/1,800-second request-timeout run sequentially.
Repeating the exact command resumes the saved schedule and skips terminal records. Do not run
another Ollama experiment or model-consuming job concurrently.

## Validate, summarize, and hash artifacts

```powershell
New-Item -ItemType Directory -Force results\reruns | Out-Null
python -m unittest discover -s tests -p test_runner_rerun.py -v *> results\reruns\pilot_test_report.txt
python analysis\validate_rerun_results.py --experiment latency --model qwen2_5_7b --pilot
python analysis\summarize_rerun.py --experiment latency --model qwen2_5_7b --pilot
python analysis\manifest_rerun.py --experiment latency --model qwen2_5_7b --pilot --test-report results\reruns\pilot_test_report.txt

python analysis\validate_rerun_results.py --experiment timeout --model qwen2_5_7b --pilot
python analysis\summarize_rerun.py --experiment timeout --model qwen2_5_7b --pilot
python analysis\manifest_rerun.py --experiment timeout --model qwen2_5_7b --pilot --test-report results\reruns\pilot_test_report.txt
```

Validation checks planned/completed/missing/unexpected IDs, duplicate records, schedule and
timeout-arm deviations, outcomes, attempt journaling, and JSONL/SQLite consistency. Summary
counts are reported separately by timeout arm and condition; incorrect answers are not counted
as infrastructure failures. Reports and manifests are write-once: choose a new `--output` path
if a report already exists. Do not treat successful execution as evidence of answer correctness.
