"""
Reader for PSO2 NGS character customization files (.fnp and friends).

Vendored from the PSO2 NGS character creator reverse engineering notes.
The parser is pure standard library and verified byte-for-byte against the
C# reference implementation (Shadowth117's PSO2-Salon-Tool CharFileLibrary).
"""

from .pso2_xxp import Blowfish, CharacterFile, load_layout

__all__ = ["Blowfish", "CharacterFile", "load_layout"]
