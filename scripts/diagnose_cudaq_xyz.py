"""Run the notebook's XYZ shots case in a monitored Linux subprocess.

Use the Python environment containing CUDA-Q. No simulations run on import.
RSS monitoring is a best-effort guard, not a hard OS memory limit.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks_MoE' / 'Benchmark_PQFMs_CPU_GPU.ipynb'


def memory_kib(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, _, value = line.partition(':')
        fields = value.split()
        if len(fields) == 2 and fields[1] == 'kB':
            values[key] = int(fields[0])
    return values


def guard_reason(rss_kib, available_kib, max_rss_gib, min_available_gib):
    if rss_kib >= max_rss_gib * 1024**2:
        return 'rss_limit'
    if available_kib <= min_available_gib * 1024**2:
        return 'low_system_memory'
    return None


def worker(directory):
    scope = {'__name__': '__diagnostic__'}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook['cells']:
        if cell.get('id') in ('configuration', 'imports', 'configuration-helpers',
                              'benchmark-definitions'):
            exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), scope)
    scope.update(QUBITS=[21], MAPS=['XYZ'], ENABLE_MPS=False,
                 ENABLED_CONFIGS=['cudaq_sv_shots'], RUN_BENCHMARK=False)
    original_make_map = scope['make_map']
    with (directory / 'stages.jsonl').open('w', buffering=1) as stages:
        def record(stage):
            stages.write(json.dumps(dict(time=time.time(), stage=stage,
                                         memory_kib=memory_kib('/proc/self/status'))) + '\n')

        def make_map(name, options):
            qfm = original_make_map(name, options)
            original_transform = qfm.transform
            calls = 0

            def transform(values):
                nonlocal calls
                label = 'warmup' if calls == 0 else f'repeat_{calls}'
                calls += 1
                record(label + '_start')
                result = original_transform(values)
                record(label + '_end')
                return result

            qfm.transform = transform
            return qfm

        scope['make_map'] = make_map
        record('start')
        results, _, _ = scope['run_benchmark'](directory)
        record('end')
    return 0 if results.status.eq('ok').all() else 1


def monitor(command, directory, max_rss_gib=8, min_available_gib=2, timeout=900):
    start = time.monotonic()
    reason = None
    peak = 0
    with (directory / 'worker.log').open('w') as log, (directory / 'memory.csv').open('w') as memory:
        writer = csv.DictWriter(memory, fieldnames=['elapsed_seconds', 'rss_kib', 'swap_kib', 'available_kib'])
        writer.writeheader()
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - start
                try:
                    usage = memory_kib(f'/proc/{process.pid}/status')
                except FileNotFoundError:
                    break
                available = memory_kib('/proc/meminfo')['MemAvailable']
                rss = usage.get('VmRSS', 0)
                peak = max(peak, rss)
                writer.writerow(dict(elapsed_seconds=elapsed, rss_kib=rss,
                                     swap_kib=usage.get('VmSwap', 0), available_kib=available))
                memory.flush()
                reason = guard_reason(rss, available, max_rss_gib, min_available_gib)
                if elapsed >= timeout:
                    reason = 'timeout'
                if reason:
                    process.kill()
                    break
                time.sleep(.2)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    report = dict(returncode=returncode, stop_reason=reason, peak_rss_kib=peak,
                  elapsed_seconds=time.monotonic() - start, command=command,
                  max_rss_gib=max_rss_gib, min_available_gib=min_available_gib)
    (directory / 'exit.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    directory = ROOT / 'notebooks_MoE/benchmark_results/diagnostics' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    directory.mkdir(parents=True)
    print(f'Diagnostic reports: {directory}', flush=True)
    report = monitor([sys.executable, '-u', str(Path(__file__).resolve()), '--worker', str(directory)], directory)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['returncode'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
