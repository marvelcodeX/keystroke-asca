"""Prompted keystroke recording session with exact key-press logging.

Shows a random, balanced, interleaved sequence of keys. You press each key when
prompted. Every key down/up is logged with wall-clock and monotonic timestamps,
giving ground-truth labels for the audio you record on the phone.

Optionally records the Mac's own mic at the same time (--mac-mic) as a second,
well-synchronised audio channel.

Usage (from keystroke-asca/):
    .venv/bin/python src/record_logger.py --keyboard macbook_air_m4 --session 1 \
        --presses-per-key 50 --distance-cm 15 --mac-mic

Needs macOS permissions for the app hosting the terminal (Terminal / iTerm / VS Code):
    System Settings > Privacy & Security > Input Monitoring  (key logging)
    System Settings > Privacy & Security > Microphone        (only with --mac-mic)
Only run this on your own machine, in your own session, with this script's prompts.
"""
import argparse
import csv
import json
import random
import sys
import termios
import threading
import time
from datetime import datetime
from pathlib import Path

KEYS = list("0123456789abcdefghijklmnopqrstuvwxyz")
ROOT = Path(__file__).resolve().parents[1]


def make_sequence(presses_per_key: int, seed: int) -> list[str]:
    """Balanced blocks: each block is a shuffle of all keys; no repeat across block edges."""
    rng = random.Random(seed)
    seq: list[str] = []
    for _ in range(presses_per_key):
        block = KEYS[:]
        rng.shuffle(block)
        while seq and block[0] == seq[-1]:
            rng.shuffle(block)
        seq.extend(block)
    return seq


def key_name(key) -> str:
    if hasattr(key, "char") and key.char is not None:
        return key.char.lower()
    return str(key).replace("Key.", "<") + ">"


class Session:
    def __init__(self, sequence, out_dir: Path, min_gap: float, max_gap: float, seed: int):
        self.sequence = sequence
        self.out_dir = out_dir
        self.min_gap, self.max_gap = min_gap, max_gap
        self.rng = random.Random(seed + 1)
        self.idx = 0
        self.target = None  # key currently being prompted (None during gap)
        self.done = threading.Event()
        self.events = []
        self.lock = threading.Lock()
        self.t0_mono = time.perf_counter()

    def log(self, kind: str, key: str):
        with self.lock:
            self.events.append(
                {
                    "t_wall": time.time(),
                    "t_mono": time.perf_counter() - self.t0_mono,
                    "event": kind,
                    "key": key,
                    "prompted": self.target or "",
                    "seq_idx": self.idx,
                }
            )

    def on_press(self, key):
        k = key_name(key)
        self.log("down", k)
        if self.target is not None and k == self.target:
            self.target = None
            self.idx += 1
            threading.Thread(target=self.next_prompt, daemon=True).start()

    def on_release(self, key):
        self.log("up", key_name(key))

    def next_prompt(self):
        if self.idx >= len(self.sequence):
            self.done.set()
            return
        time.sleep(self.rng.uniform(self.min_gap, self.max_gap))
        self.target = self.sequence[self.idx]
        sys.stdout.write(f"\r  [{self.idx + 1:>4}/{len(self.sequence)}]  press:  {self.target.upper()}   ")
        sys.stdout.flush()

    def save(self):
        path = self.out_dir / "keylog.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.events[0].keys()))
            w.writeheader()
            w.writerows(self.events)
        return path


def main():
    from pynput import keyboard  # imported lazily so tests/CI work without a keyboard backend

    ap = argparse.ArgumentParser()
    ap.add_argument("--keyboard", required=True, help="e.g. macbook_air_m4 or logitech")
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--presses-per-key", type=int, default=50)
    ap.add_argument("--distance-cm", type=float, default=15)
    ap.add_argument("--phone-position", default="left of keyboard on folded cloth")
    ap.add_argument("--min-gap", type=float, default=0.8, help="min seconds between prompts")
    ap.add_argument("--max-gap", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mac-mic", action="store_true", help="also record the Mac mic (needs sounddevice)")
    ap.add_argument("--device", default="MacBook Air Microphone", help="input device name for --mac-mic")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    out_dir = ROOT / "data" / "raw" / args.keyboard / f"session{args.session}"
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"{out_dir} already has files. Refusing to overwrite; pick a new --session.")
    out_dir.mkdir(parents=True, exist_ok=True)

    sequence = make_sequence(args.presses_per_key, seed)
    meta = {
        "keyboard": args.keyboard,
        "session": args.session,
        "created": datetime.now().isoformat(timespec="seconds"),
        "presses_per_key": args.presses_per_key,
        "distance_cm": args.distance_cm,
        "phone_position": args.phone_position,
        "seed": seed,
        "notes": args.notes,
        "mac_mic": args.mac_mic,
    }
    (out_dir / "sequence.json").write_text(json.dumps(sequence))

    print(f"\nSession: {args.keyboard} / session {args.session}  ({len(sequence)} presses)")
    print("Checklist:")
    print("  1. Phone recorder app set to WAV, 44.1/48 kHz, no noise suppression/AGC.")
    print("  2. START the phone recording now.")
    print("  3. Clap once, clearly, right after pressing ENTER below.")
    input("\nPress ENTER when the phone is recording... ")
    t_enter = time.time()
    print("Clap now, then wait 3 seconds. Prompts begin automatically.")

    mic_stream = mic_file = None
    if args.mac_mic:
        import sounddevice as sd
        import soundfile as sf

        mic_file = sf.SoundFile(out_dir / "mac_mic.wav", "w", samplerate=44100, channels=1, subtype="PCM_24")
        mic_stream = sd.InputStream(
            device=args.device, samplerate=44100, channels=1,
            callback=lambda indata, *_: mic_file.write(indata),
        )
        meta["mac_mic_start_wall"] = time.time()
        mic_stream.start()

    # Suppress terminal echo so typed keys don't clutter the prompt.
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    new_attrs = termios.tcgetattr(fd)
    new_attrs[3] &= ~(termios.ECHO | termios.ICANON)
    sess = Session(sequence, out_dir, args.min_gap, args.max_gap, seed)
    meta["enter_wall"] = t_enter
    meta["session_start_wall"] = time.time()
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
        listener = keyboard.Listener(on_press=sess.on_press, on_release=sess.on_release)
        listener.start()
        time.sleep(3)
        sess.next_prompt()
        try:
            while not sess.done.wait(0.2):
                pass
        except KeyboardInterrupt:
            print("\nInterrupted: saving what was recorded so far.")
        listener.stop()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        termios.tcflush(fd, termios.TCIFLUSH)
        if mic_stream:
            mic_stream.stop()
            mic_stream.close()
            mic_file.close()

    meta["session_end_wall"] = time.time()
    meta["presses_completed"] = sess.idx
    (out_dir / "session.json").write_text(json.dumps(meta, indent=2))
    if sess.events:
        print(f"\nSaved {sess.save()}")
    print("Now: record 5 minutes of SILENCE (keyboard untouched), then STOP the phone recording.")
    print(f"Copy the phone WAV into {out_dir}/phone.wav")


if __name__ == "__main__":
    main()
