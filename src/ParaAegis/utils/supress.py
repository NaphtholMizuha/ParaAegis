import os
import sys
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    """临时抑制所有写入 stdout 的输出。"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout