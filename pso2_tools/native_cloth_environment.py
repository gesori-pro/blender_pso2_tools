"""Live environment-field arithmetic (28C4250/28C4D30), explicit inputs.

This evaluates the actual piecewise wave, not a guessed sinusoidal wind.
Memory observation and actor velocity are separate caller responsibilities.
"""

from dataclasses import dataclass

import numpy as np

from .native_cloth_writeback import F, _array, _dot


@dataclass(frozen=True)
class EnvironmentWave:
    direction_frequency: np.ndarray
    amplitude_bias_duty: np.ndarray


def evaluate_wave(wave, position, time):
    direction = _array(wave.direction_frequency, (4,), 'Wave direction/frequency')
    params = _array(wave.amplitude_bias_duty, (4,), 'Wave amplitude/bias/duty')
    point = _array(position, (4,), 'Wave evaluation position')
    time = F(time)
    if not np.isfinite(time):
        raise ValueError('Wave time must be finite')
    frequency = direction[3]
    phase = F(F(F(frequency*F(54))*time) - _dot(direction*frequency, point))
    # ROUNDPS immediate 3 truncates towards zero, including negative phase.
    fraction = F(phase-F(np.trunc(phase)))
    shifted = F(fraction-F(F(1)-params[2]))
    value = F(0)
    if shifted > F(-.5):
        value = F(F(shifted+shifted)+F(.25)) if shifted < F(.5) else F(1)
    factor = F(F(value*params[0])+params[1])
    output_direction = direction.copy()
    output_direction[3] = F(1)
    return output_direction*factor


def evaluate_field(primary, additional, base, position, time, gain):
    """The primary wave uses time zero in the observed live helper."""
    out = _array(base, (4,), 'Base environment vector') + evaluate_wave(primary, position, F(0))
    for wave in additional:
        out = evaluate_wave(wave, position, time) + out
    gain = F(gain)
    if not np.isfinite(gain):
        raise ValueError('Environment gain must be finite')
    return out*gain


def wrapper_wind(field, response_gain, offset):
    response_gain = F(response_gain)
    if not np.isfinite(response_gain):
        raise ValueError('Wrapper environment response must be finite')
    return _array(field, (4,), 'Environment field')*response_gain + _array(offset, (4,), 'Wind offset')
