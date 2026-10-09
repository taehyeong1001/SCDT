"""Shared readout persistence and comparison with the submitted model."""
import argparse
import contextlib
import io
import json
from pathlib import Path

from common import ROOT
import numpy as np
from threadpoolctl import threadpool_limits


def train_main(system, fit, argv=None):
    parser = argparse.ArgumentParser(description=f'Train the final {system} readout on frozen sequences and matrices.')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/training')
    args = parser.parse_args(argv)
    if any(args.output.resolve().is_relative_to(ROOT / name) for name in ('models', 'data')):
        parser.error('Training outputs must not overwrite frozen assets')
    with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
        fitted, reference, protocol = fit()
    error = float(np.linalg.norm(fitted - reference) / np.linalg.norm(reference))
    output = args.output / system
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'readout.npz', W_out=fitted)
    protocol.update(relative_readout_error=error, frozen_recurrent_weights=True,
                    frozen_training_sequences=True, ODE_regenerated=False)
    (output / 'verification.json').write_text(json.dumps(protocol, indent=2) + '\n')
    if error >= 1e-7:
        raise AssertionError(f'Refit differs from frozen model: relative error={error}')
    print(f'{system}: trained readout saved to {output / "readout.npz"}; relative error={error:.3g}')
