import json
import spacy
import logging
from typing import Tuple, List, Dict
logger = logging.getLogger(__name__)

logger.debug(msg='loading ja_core_news_md')
nlp = spacy.load('ja_core_news_md')
logger.debug(msg='loading ja_core_news_md finished.')

def jopen(file_path: str):
    with open(file_path, 'r') as f:
        j = json.load(f)

    return j

def return_ner_span(text: str) -> List[Dict]:
    '''
    :param text:
    :return:
    '''
    doc = nlp(text=text)
    ents = [{'text': ent.text,
             'label': ent.label_,
             'span': (ent.start_char, ent.end_char)} for ent in doc.ents]

    return ents