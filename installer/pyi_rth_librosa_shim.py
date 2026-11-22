"""Lightweight frozen runtime shim for librosa.resample using scipy.signal.

Eliminates the ~128 MB numba/llvmlite/librosa/sklearn dependency chain that was
pulled in solely for a single `librosa.resample(..., orig_sr=sr, target_sr=24000)`
call in `holos.models.VoiceEncoder.preprocess_audio`.
"""

import importlib.machinery
import sys
import types
from fractions import Fraction


def _install_librosa_shim() -> None:
    if "librosa" in sys.modules:
        return

    mod = types.ModuleType("librosa")
    mod.__spec__ = importlib.machinery.ModuleSpec("librosa", None)
    mod.__version__ = "0.10.2"
    mod.__path__ = []

    def resample(
        y,
        *,
        orig_sr: float,
        target_sr: float,
        res_type: str = "soxr_hq",
        fix: bool = True,
        scale: bool = False,
        axis: int = -1,
        **kwargs,
    ):
        import numpy as np
        from scipy.signal import resample_poly

        arr = np.asarray(y, dtype=np.float32)
        if float(orig_sr) == float(target_sr):
            return arr
        ratio = Fraction(float(target_sr) / float(orig_sr)).limit_denominator(1000)
        resampled = resample_poly(arr, ratio.numerator, ratio.denominator, axis=axis)
        return np.asarray(resampled, dtype=np.float32)

    mod.resample = resample
    sys.modules["librosa"] = mod


_install_librosa_shim()
