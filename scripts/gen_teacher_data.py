"""Generate search-bot (teacher) games in parallel: gen_teacher_data.py START STOP OUT_DIR."""
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crownzero.datagen import worker  # noqa: E402

if __name__ == "__main__":
    start, stop = int(sys.argv[1]), int(sys.argv[2])
    out_dir = sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else mp.cpu_count()
    procs = [mp.get_context("fork").Process(target=worker, args=(i, n, start, stop, out_dir, 7))
             for i in range(n)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
