"""Explicit x64 SSE estimate provider for recovered cloth arithmetic."""

from functools import lru_cache
from pathlib import Path

import numpy as np


class HardwareEstimates:
    def __init__(self):
        # Lazy load: standalone pure math/tests do not import Blender/.NET.
        from . import dotnet
        dotnet.load()
        import clr
        assembly = Path(__file__).parent / 'bin' / 'Pso2Tools.PhysicsMath.dll'
        if not assembly.is_file():
            raise ValueError('Native physics math assembly is missing')
        clr.AddReference(str(assembly))
        from Pso2Tools.PhysicsMath import Estimate
        if not Estimate.Supported:
            raise ValueError('Native physics estimates require x64 SSE; no approximate fallback')
        self._estimate = Estimate

    @lru_cache(maxsize=4096)
    def rcp(self, value):
        return np.float32(self._estimate.Reciprocal(float(value)))

    @lru_cache(maxsize=4096)
    def rsqrt(self, value):
        return np.float32(self._estimate.ReciprocalSqrt(float(value)))
