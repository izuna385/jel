import logging

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

from jel.mention_predictor import EntityLinker

class Sentence(BaseModel):
    sentence: str

router = APIRouter()
logger = logging.getLogger(__file__)
logger.info("loading linker...")
el = EntityLinker()
logger.info("loading finished!")

@router.post("/api/v1/link")
async def link(params: Sentence):
    if params.sentence is None:
        raise HTTPException(status_code=400, detail="Sentence is required.")

    try:
        result = el.link(sentence=params.sentence)
    except Exception:
        raise HTTPException(status_code=400, detail="fail to link")

    return {"result": result}

@router.post("/api/v1/question")
async def question(params: Sentence):
    if params.sentence is None:
        raise HTTPException(status_code=400, detail="Sentence is required.")

    try:
        result = el.question(sentence=params.sentence)
    except Exception:
        raise HTTPException(status_code=400, detail="fail to link")

    return {"result": result}
