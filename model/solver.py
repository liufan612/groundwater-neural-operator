"""MODFLOW executable selected by MF6_EXE, or mf6 on PATH."""
import os
from pathlib import Path
import shutil
import subprocess


def executable():
    requested = os.environ.get('MF6_EXE', 'mf6')
    return shutil.which(requested) or requested


def run(model):
    model = Path(model)
    with (model / 'execution.log').open('w') as log:
        subprocess.run([executable()], cwd=model, stdout=log, stderr=subprocess.STDOUT,
                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), check=True)
    if 'Normal termination of simulation' not in (model/'execution.log').read_text():
        raise RuntimeError(f'MODFLOW did not terminate normally: {model}/execution.log')
