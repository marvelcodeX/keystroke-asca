import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from record_logger import KEYS, make_sequence  # noqa: E402


def test_balanced():
    seq = make_sequence(50, seed=0)
    assert len(seq) == 50 * len(KEYS)
    assert set(Counter(seq).values()) == {50}


def test_no_immediate_repeats():
    for seed in range(20):
        seq = make_sequence(10, seed=seed)
        assert all(a != b for a, b in zip(seq, seq[1:]))


def test_deterministic_per_seed():
    assert make_sequence(5, seed=3) == make_sequence(5, seed=3)
    assert make_sequence(5, seed=3) != make_sequence(5, seed=4)
