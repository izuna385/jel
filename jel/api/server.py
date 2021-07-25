import logging
from fastapi import FastAPI

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]"
)
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi import HTTPException


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from jel.mention_predictor import EntityLinker

class Sentence(BaseModel):
    sentence: str

logger = logging.getLogger(__file__)
logger.info("loading linker...")
el = EntityLinker()
logger.info("loading finished!")

@app.post("/link")
async def link(params: Sentence):
    if params.sentence is None:
        raise HTTPException(status_code=400, detail="Sentence is required.")

    try:
        result = el.link(sentence=params.sentence)
    except Exception:
        raise HTTPException(status_code=400, detail="fail to link")

    return {"result": result}

@app.post("/question")
async def question(params: Sentence):
    if params.sentence is None:
        raise HTTPException(status_code=400, detail="Sentence is required.")

    try:
        result = el.question(sentence=params.sentence)
    except Exception:
        raise HTTPException(status_code=400, detail="fail to link")

    return {"result": result}


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8000,
                log_level="debug", debug=True)