"""Sanity-check the environment: libs import, MPS available, audio devices visible."""
import importlib
import platform

for mod in ["torch", "torchaudio", "numpy", "scipy", "librosa", "soundfile", "sounddevice",
            "pynput", "sklearn", "pandas", "matplotlib", "seaborn", "yaml", "timm"]:
    try:
        m = importlib.import_module(mod)
        print(f"ok   {mod:12s} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {mod:12s} {e}")

import torch

print("\nplatform:", platform.platform(), platform.machine())
print("MPS built:", torch.backends.mps.is_built(), "| available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    x = torch.randn(256, 256, device="mps")
    print("MPS matmul ok:", (x @ x).shape)

try:
    import sounddevice as sd

    print("\nAudio input devices:")
    for d in sd.query_devices():
        if d["max_input_channels"] > 0:
            print("  ", d["name"], "| default sr", d["default_samplerate"])
except Exception as e:  # noqa: BLE001
    print("sounddevice error:", e)
