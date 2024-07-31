from logging import getLogger

from pydantic import BaseModel

logger = getLogger("clinicadl.predict_config")


class DataConfig(BaseModel):
    def __init__(self):
        print("init")


class PredictConfig(BaseModel):
    def __init__(self):
        print("init")
