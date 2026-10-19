"""Isolated one-row reproduction of the demand notebook's 30-qubit MPS benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_cudaq_xyz import monitor, memory_kib


def notebook_input():
    import numpy as np
    from sklearn.decomposition import PCA
    scope = {}
    nb = json.loads((ROOT/'notebooks_MoE/QIMED_MoE_Demanda_Pressao_pqfmlib_api.ipynb').read_text())
    for c in nb['cells']:
        if c.get('id') in ('demanda-01', 'timing-helpers', 'demanda-03', 'demanda-06', 'demanda-09'):
            exec(''.join(c['source']), scope)
    scope['DATA_DIR'] = ROOT/'notebooks_MoE/protocolo_comparacao'
    data, protocol, _ = scope['load_dataset'](scope['DATA_DIR'])
    folds = scope['make_temporal_folds'](data, protocol)
    train = data.iloc[folds[0]['inner_train']]
    columns = protocol['features_numericas'] + protocol['features_categoricas']
    values = scope['make_preprocessor'](protocol).fit_transform(train[columns])
    return PCA(n_components=29, svd_solver='full').fit_transform(values)


def worker(mode, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import numpy as np
    import torch
    import triton  # Match the notebook's LLVM/MLIR import order.
    torch.set_num_threads(1)
    from pqfmlib import HeisenbergProjectiveQFM
    with (directory/'events.jsonl').open('w', buffering=1) as stream:
        def record(stage, **kwargs):
            row = dict(stage=stage, rss_kib=memory_kib('/proc/self/status').get('VmRSS'), **kwargs)
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10)
            row['gpu_memory_mib'] = gpu.stdout.strip()
            stream.write(json.dumps(row, default=str)+'\n')
            print(json.dumps(row, default=str), flush=True)

        values = notebook_input()
        np.save(directory/'input_row.npy', values[:1])
        record('data', shape=values.shape)
        qfm = HeisenbergProjectiveQFM('mps_diagnostic', q_enc=30, R=2, alpha=.1,
                                      use_tanh_scaling=True, measure_2local_diagonal=False,
                                      framework='qiskit' if mode.startswith('qiskit') else 'cudaq',
                                      expectation_method='exact' if mode=='bond64_exact' else 'shots', mps=True, shots=4096, seed=42,
                                      mps_max_bond_dimension=64 if mode.startswith('bond64') else None,
                                      mps_truncation_threshold=0, save_circuit_drawings=False)
        start=perf_counter()
        qfm.fit(values)
        record('fit_end', seconds=perf_counter()-start)
        if not mode.startswith('qiskit'):
            import cudaq
            for name in ('get_state', 'sample'):
                original=getattr(cudaq, name)
                def wrapped(*args, _original=original, _name=name, **kwargs):
                    record(_name+'_start')
                    start=perf_counter()
                    result=_original(*args, **kwargs)
                    record(_name+'_end', seconds=perf_counter()-start)
                    return result
                setattr(cudaq, name, wrapped)
        record('transform_start')
        start=perf_counter()
        if mode == 'qiskit_exact':
            from pqfmlib.maps.heisenberg import make_heisenberg_theta_matrix
            theta, _ = make_heisenberg_theta_matrix(values[:1], 30)
            prepared = qfm._prepared_execution
            circuit = prepared.qc_t.assign_parameters(dict(zip(prepared.param_order, theta[0])))
            for i, (observable,) in enumerate(prepared.obs_isa_broadcast):
                circuit.save_expectation_value(observable, list(range(30)), label=f'obs_{i}')
            result = qfm.backend.run(circuit, shots=1).result().data()
            features = np.array([[float(result[f'obs_{i}']) for i in range(90)]])
        else:
            features=qfm.transform(values[:1])
        elapsed=perf_counter()-start
        np.save(directory/'features.npy', features)
        record('transform_end', seconds=elapsed, shape=features.shape,
               stats=getattr(qfm.backend,'last_execution_stats',None))
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['baseline','bond64','qiskit','bond64_exact','qiskit_exact'])
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--modes', nargs='+', default=['baseline','bond64','qiskit'])
    args=parser.parse_args()
    if args.worker:
        return worker(args.worker,args.directory)
    directory=ROOT/'notebooks_MoE/benchmark_results/diagnostics'/('mps30_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    directory.mkdir(parents=True)
    print(directory,flush=True)
    for mode in args.modes:
        target=directory/mode
        target.mkdir()
        result=monitor([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',mode,'--directory',str(target)], target, timeout=180)
        print(mode,json.dumps(result),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
