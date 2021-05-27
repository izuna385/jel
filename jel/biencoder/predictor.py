'''
encode mention emb from mention or entity encoder.
'''
from allennlp.common.util import JsonDict
from allennlp.predictors import Predictor
from allennlp.data import (
    Instance
)
from jel.biencoder.parameters import BiEncoderExperiemntParams
from jel.biencoder.dataset_reader import SmallJaWikiReader
import torch
from allennlp.models import Model
from jel.biencoder.model import Biencoder
from jel.biencoder.utils import vocab_loader, encoder_loader
from jel.biencoder.encoder import (
    ChiveMentionEncoder, ChiveEntityEncoder
    )
from jel.biencoder.model import Biencoder
from jel.utils.embedder import bert_emb_returner, chive_emb_returner

import numpy as np
import logging
import os
from typing import Iterable, List, Tuple
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from jel.common_config import ENCODER_DIRPATH

logger = logging.getLogger(__name__)

class MentionPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"anchor_sent": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["anchor_sent"]
        return self._dataset_reader.text_to_instance(sentence)


class EntityPredictor(Predictor):
    def predict(self, gold_title: str, gold_ent_desc: str) -> JsonDict:
        return self.predict_json({"gold_title": gold_title,
                                  "gold_ent_desc": gold_ent_desc})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        gold_title = json_dict["gold_title"]
        gold_ent_desc = json_dict["gold_ent_desc"]
        return self._dataset_reader.text_to_instance({"gold_title": gold_title,
                                                      "gold_ent_desc": gold_ent_desc})


def predictors_loader() -> Tuple[Predictor, Predictor]:
    '''
    Currently, only chive is supported.
    :return:
    '''
    params = BiEncoderExperiemntParams()
    config = params.opts
    reader = SmallJaWikiReader(config=config)

    vocab = vocab_loader(config.vocab_dir)
    embedder = chive_emb_returner(vocab=vocab)
    mention_encoder, entity_encoder = ChiveMentionEncoder(word_embedder=embedder), \
                                      ChiveEntityEncoder(word_embedder=embedder)
    mention_encoder = encoder_loader(encoder=mention_encoder,
                                     path=os.path.join(ENCODER_DIRPATH, 'mention_encoder.th'))
    entity_encoder = encoder_loader(encoder=entity_encoder,
                                     path=os.path.join(ENCODER_DIRPATH, 'entity_encoder.th'))
    model = Biencoder(config, mention_encoder, entity_encoder, vocab)


    mention_predictor = MentionPredictor(model=model,dataset_reader=reader)
    entity_predictor = EntityPredictor(model=model,dataset_reader=reader)

    return mention_predictor, entity_predictor