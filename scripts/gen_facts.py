"""Regenerate crownzero/replays/facts.py from the installed simulator."""
from pathlib import Path

from royalegym import RustEngine

rows = []
for c in RustEngine().cards():
    rows.append(f'    {c.name!r}: dict(elixir={c.elixir}, kind={(c.card_kind or "TROOP")!r}, '
                f'count={c.count}, hp={int(c.hitpoints or 0)}, flying={bool(c.flying)}, '
                f'radius={round((c.radius or 0) / 18000, 2)}),')
out = Path(__file__).resolve().parents[1] / "crownzero" / "replays" / "facts.py"
head = out.read_text().split("SIM_CARDS = {")[0]
out.write_text(head + "SIM_CARDS = {\n" + "\n".join(rows) + "\n}\n")
print(f"{len(rows)} cards -> {out}")
