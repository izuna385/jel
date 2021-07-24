import logging
from jel.api.v1 import predictor
from fastapi import FastAPI

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]"
)


def app() -> FastAPI:
    app = FastAPI()
    app.state.cache = {}
    app.include_router(predictor.router)

    return app