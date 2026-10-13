"""DAgger round: gen_dagger_data.py START STOP OUT_DIR STUDENT_MODEL [WORKERS] [BETA]"""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.dagger import worker  # noqa: E402

if __name__ == "__main__":
    start, stop, out_dir, model = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else mp.cpu_count()
    beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.15
    procs = [mp.get_context("fork").Process(target=worker,
                                            args=(i, n, start, stop, out_dir, model, 11, beta))
             for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
