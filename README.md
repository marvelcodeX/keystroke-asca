# keystroke-asca

An end-to-end ealuated study of acoustic keystroke leakage on two keyboards: a MacBook Air M4 and a Logitech external keyboard, recorded with an ordinary Android phone and classified with deep learning on Apple Silicon.


## Approach

- **Automatic ground truth.** `src/record_logger.py` prompts a balanced, shuffled key sequence and logs every key
  down/up with timestamps, so every audio clip has an exact label with no hand annotation. It can also record the Mac's
  own mic as a second, synchronised channel.
- **Honest evaluation by design.**
  - Splits are by **recording session** (train on sessions 1-2, test on a sealed session 3), never by random press, so
    neighbouring presses can't leak across the split.
  - Keys are recorded in a **random interleaved order** in one continuous take, so a model can't learn per-file noise.
  - A **silence-only control** must score at chance.
  - The best epoch is chosen on validation data; the test set is scored once, over 5 seeds, with bootstrap confidence
    intervals.
- **Keystroke detection without an oracle.** Onsets are detected from the audio alone and scored against the logged
  ground truth.
- **Features:** 64-band mel-spectrograms of a fixed window around each onset (`configs/base.yaml`).
- **Models:** classic baselines (MFCC + SVM / logistic regression), a small CNN, and CoAtNet, in PyTorch (Apple `mps`
  backend; Colab optional).

## Setup (macOS, Apple Silicon)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # exact versions used: requirements.lock.txt
.venv/bin/python src/check_env.py           # verifies libs, MPS, audio devices
.venv/bin/pytest -q
```

macOS permissions for the app running your terminal: **Input Monitoring** (key logging) and **Microphone**
(`--mac-mic`).

## Recording a session

```bash
.venv/bin/python src/record_logger.py --keyboard macbook_air_m4 --session 1 --presses-per-key 50 --mac-mic
```

Follow the prompts, then place the phone WAV at `data/raw/<keyboard>/session<N>/phone.wav`. Use `--session 0` for
throwaway trials. Session 3 is the sealed test set: do not tune on it.

Phone recording settings: WAV/PCM, 44.1 or 48 kHz, noise suppression and auto gain off, phone about 15 cm from the
keyboard on a folded cloth.

## Layout

```
configs/    experiment configs (single source of truth)
src/        recording logger, env check; segmentation/features/models/eval to come
tests/      unit tests
data/       raw audio and processed clips (git-ignored)
results/    outputs (model weights git-ignored)
```

## Ethics and responsible use

This is defensive/educational research on **the author's own keyboards and typing**. The logger is a prompted session
tool that only records keys you are asked to press, in a terminal you started, and requires explicit macOS permission.
Do not use it, or models trained with it, to record other people's typing or to recover passwords or private text.
Raw audio and key logs are git-ignored and should not be published.

## License

MIT, see [LICENSE](LICENSE).

## Inspiration

Inspired by Harrison, Toreini & Mehrnezhad (2023), [*A Practical Deep Learning-Based Acoustic Side Channel Attack on
Keyboards*](https://arxiv.org/abs/2308.01074).
