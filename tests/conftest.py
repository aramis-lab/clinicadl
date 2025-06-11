import multiprocessing as mp


def pytest_sessionstart(session):
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set in other tests
