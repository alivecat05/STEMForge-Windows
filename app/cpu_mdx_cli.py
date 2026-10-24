"""Close the disposable Windows CPU/ONNX worker after a successful CLI return.

The CPU/ONNX worker has crashed during shutdown after synchronous audio export
completed; the exact native fault is not yet identified. Exceptions and nonzero
SystemExit values still propagate, and this path runs only after main returns.
"""
import os
import sys


def finish_cli(entrypoint):
    try:
        entrypoint()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    print('[STEMForge] CPU inference returned successfully; closing worker.', flush=True)
    sys.stderr.flush()
    os._exit(0)


def main():
    from audio_separator.utils.cli import main as separator_main
    finish_cli(separator_main)
