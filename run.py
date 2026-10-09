"""Single entry point for the frozen SCDT reproduction package."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['figures', 'verify', 'evaluate', 'train', 'gt', 'test'])
    args, remaining = p.parse_known_args()
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(Path(tempfile.gettempdir()) / 'scdt-mpl'))
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    if args.command == 'test':
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v']
    elif args.command == 'train':
        train_parser = argparse.ArgumentParser()
        train_parser.add_argument('--system', choices=['food', 'power', 'kuramoto'], required=True)
        train_args, options = train_parser.parse_known_args(remaining)
        folder = dict(food='System_Food_Chain', power='System_Power_System', kuramoto='System_Kuramoto')[train_args.system]
        command = [sys.executable, '-B', str(ROOT / folder / 'train.py'), *options]
    else:
        script = dict(figures='tools/render_figures.py', verify='tools/verify.py',
                      evaluate='common/testing.py', gt='tools/generate_gt.py')[args.command]
        command = [sys.executable, '-B', str(ROOT / script), *remaining]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()
