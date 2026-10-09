"""Train Food Chain using the final frozen sequences and recurrent matrices."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import runtime as rt
from common.training import train_main


def fit():
    _, model, setup = rt.food_load()
    reference = model.W_out.copy()
    if not model.train(list(setup['sequences']), setup['normalized'], 2000):
        raise RuntimeError('Food readout training failed')
    return model.W_out, reference, dict(washout='t > 100', steps=2000, conditions=5)


if __name__ == '__main__':
    train_main('food', fit)
