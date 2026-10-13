"""Talking to the emulator: fresh screenshots, one-gesture plays, and a check that each
play really happened.

Recorded 2026-10-13, two Training Camp battles: 13 of 29 and 12 of 28 plays never
happened (the card stayed in hand). The game clock drawn on the screenshots showed why:
right before most failed plays the bot was looking at a frame 4-6 s old (e.g. "2:40 left",
then 2.1 s later "2:33"). Build-A-Bot reads the emulator's screen as a video stream in a
background thread, and that thread competes with the screen reader and the search for the
CPU and Python's interpreter lock; when it falls behind, the "latest" frame is seconds old,
until it catches up in a jump. Its detector also runs on every CPU core, which starves the
emulator itself. And every play was two separate ``adb shell input tap`` commands (card,
then tile); when the second was lost the card stayed raised and the next play misfired
(at 26.6 s the bot played a Goblin Hut and Spear Goblins came out at the left tower).

So: a screenshot is taken when the bot needs one (``screencap``, always the current
screen), the detector gets two threads, a play is one drag from the card to the tile (one
command, nothing left half-done), and every play is checked against the hand afterwards.
If drags never land on a device, plays fall back to two taps in one command.
"""
from __future__ import annotations

import io
import struct
import subprocess
import time

import numpy as np
from loguru import logger
from PIL import Image

SCREENSHOT_SIZE = (368, 652)  # what Build-A-Bot's screen reader expects
SWIPE_MS = 160  # drag duration: long enough to register as a drag, short enough to be fast
TAP_GAP_S = 0.15  # in tap mode, wait between the card tap and the tile tap

# Android PixelFormat values that screencap writes
_RAW_MODES = {1: ("RGBA", "RGBA", 4), 2: ("RGBA", "RGBX", 4), 3: ("RGB", "RGB", 3),
              5: ("RGBA", "BGRA", 4)}


def parse_screencap(data: bytes) -> Image.Image:
    """``adb exec-out screencap`` raw output -> RGB image. The header is width, height,
    pixel format (and, on newer Android, a colour space): 12 or 16 bytes."""
    if len(data) < 16:
        raise ValueError(f"screencap returned {len(data)} bytes")
    w, h, fmt = struct.unpack_from("<III", data, 0)
    if fmt not in _RAW_MODES or not (0 < w <= 8192 and 0 < h <= 8192):
        raise ValueError(f"unsupported screencap header {w}x{h} format {fmt}")
    mode, raw, bpp = _RAW_MODES[fmt]
    n = w * h * bpp
    header = len(data) - n
    if header not in (12, 16):
        raise ValueError(f"screencap size {len(data)} does not match {w}x{h}x{bpp}")
    img = Image.frombuffer(mode, (w, h), data[header:header + n], "raw", raw, 0, 1)
    return img.convert("RGB")


def use_fresh_screenshots(em_module) -> None:
    """Patch Build-A-Bot's Emulator: no video stream; ``take_screenshot`` grabs the screen
    as it is now."""
    E = em_module.Emulator
    if getattr(E, "_crownzero_capture", False):
        return

    def take_screenshot(self):
        t0 = time.perf_counter()
        cmd = [em_module.ADB_PATH, "-s", self.device_serial, "exec-out", "screencap"]
        data = subprocess.run(cmd, capture_output=True, timeout=10).stdout
        try:
            img = parse_screencap(data)
        except ValueError as e:  # some images only do PNG
            if not getattr(self, "_warned_png", False):
                logger.info(f"raw screencap unusable ({e}); using PNG screenshots")
                self._warned_png = True
            png = subprocess.run(cmd + ["-p"], capture_output=True, timeout=10).stdout
            img = Image.open(io.BytesIO(png)).convert("RGB")
        self.capture_ms = round(1000 * (time.perf_counter() - t0))
        return img.resize(SCREENSHOT_SIZE, Image.BILINEAR)

    E._start_recording = lambda self: None
    E._start_updating_frame = lambda self: None
    E.take_screenshot = take_screenshot
    E._crownzero_capture = True


def _dummy_inputs(sess, seed=0):
    rng = np.random.default_rng(seed)
    feed = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        dtype = np.float16 if "float16" in i.type else np.float32
        feed[i.name] = rng.random(shape).astype(dtype)
    return feed


def _bench(sess, feed, runs=3) -> float:
    sess.run(None, feed)  # warm-up (CoreML compiles the model on the first run)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times) // 2]


def _same(a, b) -> bool:
    for x, y in zip(a, b):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if x.shape != y.shape:
            return False
        if np.abs(x - y).mean() > 0.05 * (np.abs(y).mean() + 1e-6):
            return False
    return True


def tune_detector(onnx_module, threads=(2, 4), accelerators: bool = True) -> None:
    """Build-A-Bot's detector ran at ~670 ms a frame on a MacBook Air (recorded 2026-10-13,
    the slowest part of each decision; 160 ms on a 2-core Linux box): its model is float16,
    slow on Apple CPUs, and it never uses the Mac's GPU / Neural Engine (CoreML). When a
    model is loaded, try CoreML (where onnxruntime has it) and a few CPU thread counts on
    this machine, as it runs now (emulator included), and keep the fastest whose output
    matches the plain CPU run."""
    ort = onnx_module.ort
    if getattr(ort.InferenceSession, "_crownzero", False):
        return
    real = ort.InferenceSession

    def make(path, provs, n_threads, args, kwargs):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        return real(path, opts, *args, providers=provs, **kwargs)

    def session(path, sess_options=None, *args, providers=None, **kwargs):
        name = str(path).split("/")[-1]
        options = [(f"CPU x{n}", ["CPUExecutionProvider"], n) for n in threads]
        available = ort.get_available_providers()
        if accelerators and "CoreMLExecutionProvider" in available:
            options += [(f"CoreML {fmt}", [("CoreMLExecutionProvider", {"ModelFormat": fmt}),
                                          "CPUExecutionProvider"], threads[0])
                        for fmt in ("MLProgram", "NeuralNetwork")]
        if accelerators and "CUDAExecutionProvider" in available:
            options.append(("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"], threads[0]))
        best, best_t, reference, results = None, None, None, []
        for label, provs, n in options:
            try:
                sess = make(path, provs, n, args, kwargs)
                feed = _dummy_inputs(sess)
                out = sess.run(None, feed)
                if reference is None:
                    reference = out  # the first option is plain CPU
                elif not _same(out, reference):
                    results.append(f"{label}: output differs, skipped")
                    continue
                t = _bench(sess, feed)
            except Exception as e:  # an accelerator that cannot run this model
                results.append(f"{label}: unavailable ({str(e).splitlines()[0][:60]})")
                continue
            results.append(f"{label}: {1000 * t:.0f} ms")
            if best_t is None or t < best_t:
                best, best_t = sess, t
        logger.info(f"detector {name}: " + ", ".join(results))
        if best is None:  # nothing worked: Build-A-Bot's own way
            return real(path, sess_options, *args, providers=providers, **kwargs)
        return best

    session._crownzero = True
    ort.InferenceSession = session


limit_detector_threads = tune_detector  # the v4.3 name


def deploy(emulator, card_xy, tile_xy, mode: str) -> None:
    """One play as a single adb command. ``mode``: "swipe" (drag the card onto the tile)
    or "tap" (tap the card, then the tile)."""
    (cx, cy), (tx, ty) = card_xy, tile_xy
    if mode == "swipe":
        emulator._run_command(["shell", "input", "swipe", str(cx), str(cy), str(tx), str(ty),
                               str(SWIPE_MS)])
    else:
        emulator._run_command(["shell", f"input tap {cx} {cy}; sleep {TAP_GAP_S}; "
                                        f"input tap {tx} {ty}"])


class PlayCheck:
    """Did each play happen? A played card leaves its hand slot within a second or two."""

    LAND_S = 2.0  # still in its slot this long after the play: it did not happen
    FORGET_S = 4.0

    def __init__(self, now=None):
        self.now = now or (lambda: time.monotonic())
        self.pending: list[tuple] = []  # (time, slot, card, input mode)
        self.landed = {"swipe": 0, "tap": 0}
        self.failed = {"swipe": 0, "tap": 0}

    def played(self, slot: int, card: str, mode: str) -> None:
        self.pending.append((self.now(), slot, card, mode))

    def update(self, hand: list) -> list[tuple[str, bool]]:
        """``hand``: card names in hand now (None where unread). -> [(card, landed)] for
        the plays settled this frame."""
        t = self.now()
        out, keep = [], []
        for p in self.pending:
            t0, slot, card, mode = p
            now_card = hand[slot] if 0 <= slot < len(hand) else None
            if now_card is not None and now_card != card:
                self.landed[mode] = self.landed.get(mode, 0) + 1
                out.append((card, True))
            elif now_card == card and t - t0 >= self.LAND_S:
                self.failed[mode] = self.failed.get(mode, 0) + 1
                out.append((card, False))
            elif t - t0 < self.FORGET_S:
                keep.append(p)
        self.pending = keep
        return out

    def summary(self) -> str:
        n = sum(self.landed.values()) + sum(self.failed.values())
        return f"{sum(self.landed.values())}/{n} plays landed"
