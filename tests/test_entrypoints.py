from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.testing import resolve_readout


class EntryPointTests(unittest.TestCase):
    def test_default_ignores_training_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch('common.testing.ROOT', root):
                for system in ('food', 'power', 'kuramoto'):
                    with self.subTest(system=system):
                        self.assertIsNone(resolve_readout(system))
                        trained = root / 'outputs/training' / system / 'readout.npz'
                        trained.parent.mkdir(parents=True)
                        trained.touch()
                        self.assertIsNone(resolve_readout(system))
                        self.assertIsNone(resolve_readout(system, frozen=True))
                        self.assertEqual(resolve_readout(system, explicit=trained), trained)

    def test_explicit_readout_must_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            readout = Path(folder) / 'weights.npz'
            with self.assertRaises(ValueError):
                resolve_readout('kuramoto', explicit=readout)
            readout.touch()
            self.assertEqual(resolve_readout('kuramoto', explicit=readout), readout)
