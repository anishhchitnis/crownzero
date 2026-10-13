"""Record one battle (search bot as Blue) and write a self-contained replay page."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from royalegym import ReplayRecorder, save_trace, verify_trace  # noqa: E402
from royalegym.render import write_html  # noqa: E402

from crownzero.arena import Scripted, make_env, play_game  # noqa: E402
from crownzero.search_bot import SearchBot  # noqa: E402

opp = sys.argv[1] if len(sys.argv) > 1 else "defend"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7

env = make_env()
env.recorder = ReplayRecorder(frame_every_tick=True)
winner, ticks, crowns, _ = play_game(SearchBot(), Scripted(opp), seed=seed)
trace = env.recorder.trace
env.recorder = None

out = ROOT / "replays"
out.mkdir(exist_ok=True)
save_trace(trace, out / f"search_vs_{opp}_seed{seed}.msgpack")
problems = verify_trace(trace, make_env().engine)
page = write_html(trace, out / f"search_vs_{opp}_seed{seed}.html",
                  title=f"Search bot (blue) vs {opp} - seed {seed}", stride=2)
print(f"winner {winner} crowns {crowns} ticks {ticks} | replay re-runs identically: "
      f"{not problems} | {page}")
