"""Record replays from the emulator, as training data for learning from top players.

    python -m crownzero.real.capture --device-serial 127.0.0.1:5555 --name top_ladder_1

Open a replay in the game (TV Royale, or a top player's battle log), start this, let the
replay play to the end, then press Ctrl+C. The video is recorded on the emulator itself
(``adb screenrecord``, clean frames at the device's own resolution, no window chrome),
in chunks of a bit under 3 minutes because that is screenrecord's limit, then copied to
``--out/<name>_<date-time>/part_000.mp4, part_001.mp4, ...`` with a meta.json.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import time
from pathlib import Path

CHUNK_S = 170


def find_adb() -> str:
    try:  # the ADB Build-A-Bot downloaded
        from clashroyalebuildabot.constants import ADB_PATH
        if Path(ADB_PATH).exists():
            return ADB_PATH
    except Exception:
        pass
    adb = shutil.which("adb")
    if not adb:
        raise SystemExit("ADB not found: run the bot once (it downloads ADB) or install it")
    return adb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-serial", default="127.0.0.1:5555")
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "cr_replays"))
    ap.add_argument("--name", default="replay")
    args = ap.parse_args()

    adb = find_adb()
    s = args.device_serial

    def run(*cmd, check=False, **kw):
        return subprocess.run([adb, "-s", s, *cmd], capture_output=True, text=True,
                              check=check, **kw)

    if ":" in s:
        subprocess.run([adb, "connect", s], capture_output=True, text=True)
    size = run("shell", "wm", "size").stdout.strip()
    print(f"device: {s} | {size}")
    if "720x1280" not in size:
        print("warning: the screen is not 720x1280; the screen reader expects that size")

    out = Path(args.out).expanduser() / f"{args.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(dict(device=s, wm_size=size, name=args.name,
                                                   started=time.strftime("%c")), indent=2))
    print(f"recording to {out}  (Ctrl+C when the replay ends)")

    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))
    part = 0
    while not stop["now"]:
        remote = f"/sdcard/crz_part_{part:03d}.mp4"
        proc = subprocess.Popen([adb, "-s", s, "shell", "screenrecord", "--time-limit",
                                 str(CHUNK_S), "--bit-rate", "6000000", remote],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)  # Ctrl+C must not kill it mid-file
        t0 = time.time()
        while proc.poll() is None and not stop["now"]:
            time.sleep(0.2)
        if stop["now"] and proc.poll() is None:
            run("shell", "pkill", "-INT", "screenrecord")  # finishes the mp4 cleanly
            proc.wait(timeout=10)
            time.sleep(1.0)
        local = out / f"part_{part:03d}.mp4"
        run("pull", remote, str(local))
        run("shell", "rm", "-f", remote)
        took = time.time() - t0
        mb = local.stat().st_size / 1e6 if local.exists() else 0
        print(f"part {part}: {took:.0f}s, {mb:.1f} MB")
        part += 1
    print(f"done: {part} part(s) in {out}")


if __name__ == "__main__":
    main()
