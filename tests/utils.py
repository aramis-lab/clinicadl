import multiprocessing as mp
import platform


def safe_multiprocessing():
    if platform.system() == "Darwin":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
