"""Compare XYZ transform with direct CUDA-Q observe, in separate monitored processes.

The direct case reuses the same converted kernel, observables and input angles.
This isolates recurring integration overhead, not a different circuit synthesis.
Run with the CUDA-Q environment: .venv_gpu/bin/python scripts/compare_cudaq_xyz.py
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.diagnose_cudaq_xyz import NOTEBOOK, memory_kib, monitor


def bind_angles(compiled, param_order, theta):
    positions = {p.name: i for i, p in enumerate(param_order)}
    return [[float(expression.bind({p: values[positions[p.name]]
                                    for p in expression.parameters}))
             for expression in compiled.expressions] for values in theta]


def direct_batch(cudaq, compiled, angles, shots, seed, aggregate=None):
    """Same observe calls and output reconstruction as production; no retained results."""
    import numpy as np
    cudaq.set_random_seed(seed)
    features = np.zeros((len(angles), len(compiled.observable_terms)))
    observe_seconds = 0.0
    for row, values in enumerate(angles):
        start = perf_counter()
        results = cudaq.observe(compiled.kernel,
                                compiled.observables if aggregate is None else aggregate[0],
                                values, shots_count=shots)
        observe_seconds += perf_counter() - start
        expectations = ([result.expectation() for result in results] if aggregate is None
                        else [results.expectation(term) for term in aggregate[1]])
        features[row] = [sum(coefficient * (1.0 if index == -1 else expectations[index])
                             for index, coefficient in terms)
                         for terms in compiled.observable_terms]
    return features, observe_seconds


def load_scope():
    scope = {'__name__': '__comparison__'}
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell.get('id') in ('configuration', 'imports', 'configuration-helpers', 'benchmark-definitions'):
            exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), scope)
    scope.update(QUBITS=[21], MAPS=['XYZ'], ENABLE_MPS=False,
                 ENABLED_CONFIGS=['cudaq_sv_shots'], BENCHMARK_SAMPLES=32,
                 BENCHMARK_REPEATS=3, SHOTS=4096, RUN_BENCHMARK=False)
    return scope


def worker(mode, directory):
    import numpy as np
    import pqfmlib.core.cudaq_execution as execution_module

    scope = load_scope()
    configuration = scope['validate_settings']()[0]
    scope['preflight'](configuration)
    fit, evaluation, _ = scope['prepare_data'](21)
    options = scope['map_options']('XYZ', 21, configuration)
    metadata = scope['environment_metadata']()
    metadata.update(mode=mode, options=options, samples=32, repeats=3,
                    direct_definition='Same converted kernel; angles pre-bound outside batch timing')
    (directory / 'environment.json').write_text(json.dumps(metadata, indent=2, default=str))
    with (directory / 'events.jsonl').open('w', buffering=1) as log:
        def record(stage, **values):
            entry = dict(stage=stage, rss_kib=memory_kib('/proc/self/status').get('VmRSS', 0), **values)
            log.write(json.dumps(entry, default=str) + '\n')
            print(json.dumps(entry, default=str), flush=True)

        # Instrument the existing conversion without changing its implementation.
        original_compile = execution_module._compile_job

        def timed_compile(*args, **kwargs):
            start = perf_counter()
            result = original_compile(*args, **kwargs)
            record('conversion', seconds=perf_counter() - start)
            return result

        execution_module._compile_job = timed_compile
        start = perf_counter()
        qfm = scope['make_map']('XYZ', options)
        qfm.fit(fit)
        record('fit', seconds=perf_counter() - start)
        scope['effective_backend'](qfm, configuration)
        prepared = qfm._prepared_execution
        executor = qfm.backend
        record('prepared', observables=len(prepared.obs_isa_broadcast),
               circuit_operations=dict(prepared.qc_t.count_ops()), depth=prepared.qc_t.depth())

        aggregate = None
        if mode in ('direct', 'aggregate'):
            # Obtain exactly the production theta matrix, but intercept before simulation.
            captured = []
            original_execute = executor.execute

            def capture(prepared, theta, **kwargs):
                captured.append(np.asarray(theta).copy())
                return np.zeros((len(theta), len(prepared.obs_isa_broadcast))), prepared.obs_metadata

            executor.execute = capture
            start = perf_counter()
            try:
                qfm.transform(evaluation)
            finally:
                executor.execute = original_execute
            record('input_preparation', seconds=perf_counter() - start)
            with executor._target_context(21):
                compiled = timed_compile(executor._cudaq, prepared)
            start = perf_counter()
            angles = bind_angles(compiled, prepared.param_order, captured[0])
            record('bind_batch', seconds=perf_counter() - start)
            if mode == 'aggregate':
                # Same unit-weight distinct terms that list-observe internally sums.
                operator = executor._cudaq.SpinOperator.empty()
                terms = []
                for observable in compiled.observables:
                    canonical = observable.canonicalize()
                    if canonical.term_count != 1:
                        raise ValueError('Diagnostic requires one Pauli term per observable.')
                    operator += canonical
                    terms.append(canonical if isinstance(canonical, executor._cudaq.SpinOperatorTerm)
                                 else next(iter(canonical)))
                aggregate = (operator, terms)
            # The measured loop below does not use fit, transform or executor.execute.

        for repeat in range(4):
            stage = 'warmup' if repeat == 0 else f'repeat_{repeat}'
            rows = evaluation[:1] if repeat == 0 else evaluation
            record(stage + '_start')
            start = perf_counter()
            if mode == 'adapter':
                features = qfm.transform(rows)
                elapsed = perf_counter() - start
                stats = dict(executor.last_execution_stats)
            else:
                with executor._target_context(21):
                    features, observe_seconds = direct_batch(
                        executor._cudaq, compiled, angles[:1] if repeat == 0 else angles,
                        scope['SHOTS'], scope['SEED'], aggregate=aggregate)
                elapsed = perf_counter() - start
                stats = dict(observe_seconds=observe_seconds, observe_calls=len(rows))
            if features.shape != (len(rows), len(prepared.obs_isa_broadcast)) or not np.isfinite(features).all():
                raise RuntimeError('Invalid output shape or non-finite features.')
            record(stage + '_end', seconds=elapsed, stats=stats)
            np.save(directory / f'{stage}_features.npy', features)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['adapter', 'direct', 'aggregate'])
    parser.add_argument('--modes', nargs='+', choices=['adapter', 'direct', 'aggregate'],
                        default=['adapter', 'direct'])
    parser.add_argument('--directory', type=Path)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker, args.directory)
    directory = ROOT / 'notebooks_MoE/benchmark_results/diagnostics' / ('compare_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    directory.mkdir(parents=True)
    print(f'Reports: {directory}', flush=True)
    reports = {}
    for mode in args.modes:
        target = directory / mode
        target.mkdir()
        reports[mode] = monitor([sys.executable, '-u', str(Path(__file__).resolve()),
                                 '--worker', mode, '--directory', str(target)], target)
        print(mode, json.dumps(reports[mode]), flush=True)
    (directory / 'comparison_exit.json').write_text(json.dumps(reports, indent=2))
    return int(any(report['returncode'] != 0 for report in reports.values()))


if __name__ == '__main__':
    raise SystemExit(main())
