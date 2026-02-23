#!/usr/bin/env python3
"""
Benchmark wrapper for running a command and sampling resource usage.

This module provides a library function and a small CLI to run an external
command while periodically sampling:

- process CPU percent (aggregate across process + children)
- process memory RSS (bytes)
- system CPU percent
- GPU utilization and GPU memory (via `nvidia-smi`, optional)

It writes a CSV of raw samples and prints a small summary with min/avg/median/max
for each metric. It is designed to be a library (callable) and a simple CLI
wrapper that can be used to benchmark the OMERO inference command you showed:

Example:
    time PYTHONPATH=src python3 src/shell/benchmark.py \
        --interval 0.5 \
        --csv ./bench_samples.csv \
        --json ./bench_summary.json \
        -- timeout PYTHONPATH=src python3 src/shell/cli.py infer-omero \
            --host wss://wsi.lavlab.mcw.edu/omero-wss --port 443 \
            --username mjbarrett --password gzyxby01 \
            --output ./test.png --image-id 512

Notes:
- The CLI uses "--" (or "timeout" / "time" naming) to separate the wrapper's
  flags from the command to benchmark. You can also call the library function
  directly from Python for programmatic use.
- Requires `psutil` for process/system metrics. GPU metrics are retrieved via
  `nvidia-smi` (if available on PATH). Code gracefully falls back if either
  facility isn't present.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import psutil
except Exception:  # pragma: no cover - runtime environment may lack psutil
    psutil = None  # type: ignore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_macos_static_gpu_info() -> Optional[Dict[str, Any]]:
    """Get static GPU info on macOS without sudo via system_profiler.

    Returns a dict with keys (model, vram_mib) or None if not available.
    """
    try:
        if platform.system() != "Darwin":
            return None
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        j = json.loads(out.decode("utf8", errors="replace"))
        displays = j.get("SPDisplaysDataType") or []
        if not displays:
            return None
        d0 = displays[0]
        model = d0.get("sppci_model") or d0.get("_name") or d0.get("sppci_vendor")
        vram = d0.get("spdisplays_vram") or d0.get("spdisplays_vram_shared")
        vram_mib: Optional[int] = None
        if isinstance(vram, str):
            parts = vram.split()
            try:
                val = float(parts[0])
                unit = parts[1].upper() if len(parts) > 1 else "MB"
                if "GB" in unit:
                    vram_mib = int(val * 1024)
                else:
                    vram_mib = int(val)
            except Exception:
                vram_mib = None
        return {"model": model, "vram_mib": vram_mib}
    except Exception:
        return None


def _process_uses_metal(pid: int) -> Optional[bool]:
    """Heuristic: check whether process has opened Metal framework files (no sudo).

    Uses `lsof -p PID` (no sudo typically) and looks for substrings like
    'Metal.framework' or 'Metal' in the list of opened files. Returns True/False
    or None if the check isn't available.
    """
    try:
        if platform.system() != "Darwin":
            return None
        if shutil.which("lsof") is None:
            return None
        out = subprocess.check_output(
            ["lsof", "-p", str(pid)], stderr=subprocess.DEVNULL, timeout=1.0
        )
        s = out.decode("utf8", errors="ignore")
        if "Metal.framework" in s or "Metal" in s or "AGX" in s:
            return True
        return False
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def now_s() -> float:
    """Return monotonic timestamp (seconds)."""
    return time.perf_counter()


def safe_run_nvidia_smi() -> Optional[str]:
    """Return raw nvidia-smi output line (CSV) or None if not available.

    The output format is:
        <utilization.gpu>,<memory.used>,<memory.total>\n

    We run nvidia-smi with a very small query to avoid expensive output.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        # Query GPU utilization (%) and memory used/total (MiB) in CSV, no header.
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("utf8", errors="replace").strip()
    except Exception:
        return None


def parse_nvidia_smi(
    csv_line: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse a single nvidia-smi CSV line (first GPU) to floats.

    Returns (gpu_util_pct, gpu_mem_used_mib, gpu_mem_total_mib).
    If parsing fails, returns (None, None, None).
    """
    try:
        # nvidia-smi returns one line per GPU. We use the first GPU's metrics.
        first = csv_line.splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        if len(parts) < 3:
            return None, None, None
        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
        return gpu_util, mem_used, mem_total
    except Exception:
        return None, None, None


def aggregate_stats(values: Iterable[float]) -> Dict[str, Optional[float]]:
    """Compute min/avg/median/max for a non-empty sequence; handle empty."""
    vals = list(v for v in values if v is not None)
    if not vals:
        return {"min": None, "avg": None, "median": None, "max": None}
    return {
        "min": float(min(vals)),
        "avg": float(statistics.mean(vals)),
        "median": float(statistics.median(vals)),
        "max": float(max(vals)),
    }


# ---------------------------------------------------------------------------
# Core benchmarking function
# ---------------------------------------------------------------------------


def benchmark_command(
    cmd: List[str],
    *,
    sampling_interval: float = 0.5,
    csv_path: Optional[str] = None,
    json_path: Optional[str] = None,
    timeout: Optional[float] = None,
    capture_stdout: bool = False,
    capture_stderr: bool = False,
) -> Dict[str, Any]:
    """Run `cmd` (list[str]) and sample resource usage until completion.

    Parameters
    ----------
    cmd:
        Command to run as list (no shell). Example: ["python3", "script.py", "arg"].
    sampling_interval:
        Seconds between samples (float). Default 0.5.
    csv_path:
        Optional path to write raw sample CSV.
    json_path:
        Optional path to write summary JSON.
    timeout:
        Optional timeout (seconds) after which the process will be terminated.
    capture_stdout, capture_stderr:
        Whether to capture stdout/stderr and include them in the result (careful
        with large outputs).

    Returns
    -------
    A dictionary with:
      - timing: wall_seconds, exit_code, timed_out
      - samples: list of sample dicts (if csv_path not set this may still be present)
      - summary: aggregated metric stats
      - (optionally) stdout/stderr as strings
    """
    if sampling_interval <= 0:
        raise ValueError("sampling_interval must be > 0")

    if psutil is None:
        raise RuntimeError(
            "psutil is required for benchmarking. Install it with: pip install psutil"
        )

    # Launch the child process
    popen_kwargs = {}
    if capture_stdout:
        popen_kwargs["stdout"] = subprocess.PIPE
    if capture_stderr:
        popen_kwargs["stderr"] = subprocess.PIPE

    proc = subprocess.Popen(cmd, **popen_kwargs)
    proc_ps = psutil.Process(proc.pid)

    # Collect samples as list of dicts
    samples: List[Dict[str, Any]] = []
    start_time = now_s()
    last_cpu_times = None  # type: Optional[float]
    last_sample_time = start_time

    # Initialize CPU times baseline
    try:
        last_cpu_times = sum(proc_ps.cpu_times()[:2])
    except Exception:
        last_cpu_times = 0.0

    timed_out = False
    # Precheck for nvidia-smi availability
    nvsmi_available = shutil.which("nvidia-smi") is not None

    # Sampling loop
    try:
        while True:
            # Check process status
            if proc.poll() is not None:
                # process finished; take one final sample then break
                tstamp = now_s()
                try:
                    cpu_times = sum(proc_ps.cpu_times()[:2])
                except Exception:
                    cpu_times = last_cpu_times or 0.0
                elapsed = (
                    tstamp - last_sample_time
                    if last_sample_time is not None
                    else sampling_interval
                )
                # process CPU percent computed as delta CPU time / wall time / num_cpus * 100
                try:
                    cpu_count = psutil.cpu_count(logical=True) or 1
                except Exception:
                    cpu_count = 1
                proc_cpu_pct = 0.0
                if last_cpu_times is not None and elapsed > 0:
                    proc_cpu_pct = (
                        100.0
                        * ((cpu_times - last_cpu_times) / elapsed)
                        / max(1, cpu_count)
                    )
                last_cpu_times = cpu_times

                # process + children mem (RSS)
                rss = 0
                try:
                    rss = proc_ps.memory_info().rss
                    for child in proc_ps.children(recursive=True):
                        try:
                            rss += child.memory_info().rss
                        except Exception:
                            pass
                except Exception:
                    rss = 0

                sys_cpu = psutil.cpu_percent(interval=None)
                gpu_line = safe_run_nvidia_smi() if nvsmi_available else None
                gpu_util, gpu_mem_used, gpu_mem_total = (None, None, None)
                if gpu_line:
                    gpu_util, gpu_mem_used, gpu_mem_total = parse_nvidia_smi(gpu_line)

                metal_flag = (
                    _process_uses_metal(proc.pid)
                    if platform.system() == "Darwin"
                    else None
                )
                mac_info = (
                    _get_macos_static_gpu_info()
                    if platform.system() == "Darwin"
                    else None
                )
                samples.append(
                    {
                        "t": tstamp - start_time,
                        "proc_cpu_pct": proc_cpu_pct,
                        "proc_rss_bytes": int(rss),
                        "system_cpu_pct": float(sys_cpu),
                        "gpu_util_pct": gpu_util,
                        "gpu_mem_used_mib": gpu_mem_used,
                        "gpu_mem_total_mib": gpu_mem_total,
                        "mac_gpu_model": mac_info.get("model") if mac_info else None,
                        "mac_gpu_vram_mib": mac_info.get("vram_mib")
                        if mac_info
                        else None,
                        "mac_process_uses_metal": bool(metal_flag)
                        if metal_flag is not None
                        else None,
                    }
                )
                break

            # Regular sample
            tstamp = now_s()
            try:
                cpu_times = sum(proc_ps.cpu_times()[:2])
            except Exception:
                cpu_times = last_cpu_times or 0.0

            # compute CPU fraction since last sample
            wall_dt = (
                tstamp - last_sample_time
                if last_sample_time is not None
                else sampling_interval
            )
            try:
                cpu_count = psutil.cpu_count(logical=True) or 1
            except Exception:
                cpu_count = 1

            proc_cpu_pct = 0.0
            if last_cpu_times is not None and wall_dt > 0:
                proc_cpu_pct = (
                    100.0 * ((cpu_times - last_cpu_times) / wall_dt) / max(1, cpu_count)
                )
            last_cpu_times = cpu_times
            last_sample_time = tstamp

            # Memory (process + children)
            rss = 0
            try:
                rss = proc_ps.memory_info().rss
                for child in proc_ps.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except Exception:
                        pass
            except Exception:
                rss = 0

            # System CPU (instant)
            sys_cpu = psutil.cpu_percent(interval=None)

            # GPU (first GPU line)
            gpu_line = safe_run_nvidia_smi() if nvsmi_available else None
            gpu_util, gpu_mem_used, gpu_mem_total = (None, None, None)
            if gpu_line:
                gpu_util, gpu_mem_used, gpu_mem_total = parse_nvidia_smi(gpu_line)

            metal_flag = (
                _process_uses_metal(proc.pid) if platform.system() == "Darwin" else None
            )
            mac_info = (
                _get_macos_static_gpu_info() if platform.system() == "Darwin" else None
            )
            samples.append(
                {
                    "t": tstamp - start_time,
                    "proc_cpu_pct": float(proc_cpu_pct),
                    "proc_rss_bytes": int(rss),
                    "system_cpu_pct": float(sys_cpu),
                    "gpu_util_pct": gpu_util,
                    "gpu_mem_used_mib": gpu_mem_used,
                    "gpu_mem_total_mib": gpu_mem_total,
                    "mac_gpu_model": mac_info.get("model") if mac_info else None,
                    "mac_gpu_vram_mib": mac_info.get("vram_mib") if mac_info else None,
                    "mac_process_uses_metal": bool(metal_flag)
                    if metal_flag is not None
                    else None,
                }
            )

            # Check timeout
            if timeout is not None and (tstamp - start_time) >= float(timeout):
                try:
                    proc.terminate()
                    timed_out = True
                except Exception:
                    pass
                # allow a short grace period then kill
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                break

            # Sleep until next sample (but wake up early if process exits)
            # Use small sleeps to be responsive to termination while maintaining approximate interval.
            slept = 0.0
            while slept < sampling_interval:
                if proc.poll() is not None:
                    break
                to_sleep = min(0.1, sampling_interval - slept)
                time.sleep(to_sleep)
                slept += to_sleep

    finally:
        # ensure subprocess pipes are read if captured
        out_str = None
        err_str = None
        if capture_stdout and proc.stdout is not None:
            try:
                out_str = proc.stdout.read().decode("utf8", errors="replace")
            except Exception:
                out_str = None
        if capture_stderr and proc.stderr is not None:
            try:
                err_str = proc.stderr.read().decode("utf8", errors="replace")
            except Exception:
                err_str = None

    end_time = now_s()
    wall_seconds = end_time - start_time
    exit_code = proc.returncode

    # Write CSV of samples
    if csv_path:
        header = [
            "t",
            "proc_cpu_pct",
            "proc_rss_bytes",
            "system_cpu_pct",
            "gpu_util_pct",
            "gpu_mem_used_mib",
            "gpu_mem_total_mib",
            "mac_gpu_model",
            "mac_gpu_vram_mib",
            "mac_process_uses_metal",
        ]

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for s in samples:
                    writer.writerow({k: s.get(k) for k in header})
        except Exception as exc:
            print(f"Warning: failed to write CSV {csv_path}: {exc}", file=sys.stderr)

    # Compute summary stats
    proc_cpu_vals = [
        s["proc_cpu_pct"] for s in samples if s.get("proc_cpu_pct") is not None
    ]
    proc_rss_vals = [
        s["proc_rss_bytes"] for s in samples if s.get("proc_rss_bytes") is not None
    ]
    sys_cpu_vals = [
        s["system_cpu_pct"] for s in samples if s.get("system_cpu_pct") is not None
    ]
    gpu_util_vals = [
        s["gpu_util_pct"] for s in samples if s.get("gpu_util_pct") is not None
    ]
    gpu_mem_used_vals = [
        s["gpu_mem_used_mib"] for s in samples if s.get("gpu_mem_used_mib") is not None
    ]

    summary = {
        "proc_cpu_pct": aggregate_stats(proc_cpu_vals),
        "proc_rss_bytes": aggregate_stats(proc_rss_vals),
        "system_cpu_pct": aggregate_stats(sys_cpu_vals),
        "gpu_util_pct": aggregate_stats(gpu_util_vals),
        "gpu_mem_used_mib": aggregate_stats(gpu_mem_used_vals),
        "wall_seconds": float(wall_seconds),
        "exit_code": exit_code,
        "timed_out": bool(timed_out),
    }

    result: Dict[str, Any] = {
        "timing": {
            "wall_seconds": float(wall_seconds),
            "exit_code": exit_code,
            "timed_out": bool(timed_out),
        },
        "samples": samples,
        "summary": summary,
    }
    if capture_stdout:
        result["stdout"] = out_str
    if capture_stderr:
        result["stderr"] = err_str

    if json_path:
        try:
            with open(json_path, "w") as f:
                json.dump(result, f, indent=2, default=lambda o: str(o))
        except Exception as exc:
            print(f"Warning: failed to write JSON {json_path}: {exc}", file=sys.stderr)

    # Print a concise summary to stdout
    print("Benchmark summary:")
    print(f"  wall_seconds: {wall_seconds:.3f}")
    print(f"  exit_code: {exit_code}  timed_out: {timed_out}")

    def fmt_stats(label: str, data: Dict[str, Optional[float]], unit: str = "") -> None:
        if data["avg"] is None:
            print(f"  {label}: (no data)")
        else:
            print(
                f"  {label}: avg={data['avg']:.3f}{unit} med={data['median']:.3f}{unit} "
                f"min={data['min']:.3f}{unit} max={data['max']:.3f}{unit}"
            )

    fmt_stats("proc_cpu_pct", summary["proc_cpu_pct"], "%")
    fmt_stats("proc_rss_bytes", summary["proc_rss_bytes"], " B")
    fmt_stats("system_cpu_pct", summary["system_cpu_pct"], "%")
    fmt_stats("gpu_util_pct", summary["gpu_util_pct"], "%")
    fmt_stats("gpu_mem_used_mib", summary["gpu_mem_used_mib"], " MiB")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="shell-benchmark",
        description="Run a command while sampling CPU / RAM / GPU usage and summarise results.",
    )
    p.add_argument(
        "--interval", type=float, default=0.5, help="Sampling interval (seconds)."
    )
    p.add_argument("--csv", type=str, default=None, help="Write raw samples to CSV.")
    p.add_argument("--json", type=str, default=None, help="Write full summary to JSON.")
    p.add_argument(
        "--timeout", type=float, default=None, help="Kill command after N seconds."
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Capture stdout and include in JSON output.",
    )
    p.add_argument(
        "--stderr",
        action="store_true",
        help="Capture stderr and include in JSON output.",
    )
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help=(
            "The command to run. Use '--' before the command if you want to pass "
            "options that look like the wrapper's flags. Example: -- python3 foo.py arg"
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.cmd:
        print("No command provided. See --help.", file=sys.stderr)
        return 2

    # If the first token is '--' strip it (convenience)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    # If the command is provided as a single shell-like string, shlex.split it.
    if len(cmd) == 1:
        cmd = shlex.split(cmd[0])

    print("Benchmark wrapper starting:")
    print("  command:", " ".join(shlex.quote(c) for c in cmd))
    print(f"  sampling interval: {args.interval}s")
    if args.csv:
        print(f"  writing CSV -> {args.csv}")
    if args.json:
        print(f"  writing JSON -> {args.json}")

    try:
        res = benchmark_command(
            cmd,
            sampling_interval=args.interval,
            csv_path=args.csv,
            json_path=args.json,
            timeout=args.timeout,
            capture_stdout=args.stdout,
            capture_stderr=args.stderr,
        )
    except Exception as exc:
        print(f"Error while benchmarking: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
