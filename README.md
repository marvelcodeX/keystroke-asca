# keystroke-asca

A careful re-examination of the acoustic keystroke side channel on laptop and external keyboards, inspired by
[Harrison, Toreini & Mehrnezhad (2023), *A Practical Deep Learning-Based Acoustic Side Channel Attack on Keyboards*](https://arxiv.org/abs/2308.01074).

Goal: train classifiers on keystroke audio from a MacBook Air M4 keyboard and a Logitech keyboard, recorded with an
Android phone, and measure how well keys can be identified **under evaluation that avoids the paper's leakage problems**.

> **Status:** early. Environment, recording/labelling tool and config are done. Segmentation, features, models and
> evaluation are next. No results yet.

## Why not just replicate the paper?

| Issue in the paper | What this project does instead |
|---|---|
| Random train/test split of presses from the same session (leakage; their own Zoom numbers swing 0.52 to 0.96 on split type) | Split by **recording session** (train on sessions 1-2, test on sealed session 3) |
| Each key recorded as its own 25-press block (model can learn per-file noise) | **Random interleaved** prompted sequence in one continuous take, plus a **silence-only control** that must score at chance |
| "Peak validation accuracy" reported, 180-sample test set, single seed | Pick epoch on validation, evaluate test once; 5 seeds; bootstrap confidence intervals |
| Training collapsed at LR 1e-3 | AdamW, warmup + cosine, gradient clipping, early stopping |
| Keystroke detector needs the true press count | Detector without an oracle count, scored against logged ground truth |
| Inconsistent spectrogram parameters | One config file, logged per run |
| No simple baselines | MFCC + SVM/logistic regression, small CNN, then CoAtNet |

## Method in brief

- **Ground truth:** `src/record_logger.py` prompts a balanced random key sequence and logs every key down/up with
  timestamps (plus optional Mac-mic recording), so audio is labelled automatically.
- **Audio:** phone WAV (44.1/48 kHz, no noise suppression) about 15 cm from the keyboard.
- **Features:** 64-band mel-spectrograms of a fixed window around each onset (`configs/base.yaml`).
- **Models:** baselines then CNN / CoAtNet in PyTorch (Apple `mps` backend; Colab optional).

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

## Citation

Harrison, J., Toreini, E., Mehrnezhad, M. (2023). *A Practical Deep Learning-Based Acoustic Side Channel Attack on
Keyboards.* arXiv:2308.01074.

## License

MIT, see [LICENSE](LICENSE).
