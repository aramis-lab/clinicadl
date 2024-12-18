BATCH_SIZE = 1

SAMPLING_WEIGHTS = None  # no weighted sampling

SHUFFLE = False
DROP_LAST = False

NUM_WORKERS = 0  # main process loads data
PREFETCH_FACTOR = None

PIN_MEMORY = True  # training is supposed to be on a GPU

DP_DEGREE = None  # no data parallelism
RANK = None
