from jel.kb import TitleIndexerWithFaiss
from jel.biencoder.predictor import predictors_loader
from jel.utils.common import return_ner_span
import pdb
import os
from jel.common_config import PRIOR_DICT_PATH
import json
from typing import List, Tuple, Dict
import numpy as np
from jel.file_cache import resource_downloader
from jel.common_config import CACHE_ROOT
import logging
logger = logging.getLogger(__name__)


class EntityLinker:
    def __init__(self):
        if not os.path.exists(str(CACHE_ROOT)+'/resources.zip'):
            print('Downloading predictor. This might take few minutes.')
            resource_downloader()
        print('Loading predictor. This might take few minutes.')
        self.mention_predictor, _ = predictors_loader()
        print('Loading kb. This might take few minutes.')
        self.kb = TitleIndexerWithFaiss()
        self.prior_dict = self._mention2cand_entity_dict_loader()
        self.candidate_ent_max_num = 10

    def link(self, sentence: str):
        ne_spans = return_ner_span(text=sentence)
        for predicted_ne in ne_spans:
            mention = predicted_ne['text']
            span_start = predicted_ne['span'][0]
            span_end = predicted_ne['span'][1]

            split_strings = list(sentence)
            split_strings.insert(span_start, '<a>')
            split_strings.insert(span_end + 1, '</a>')
            anchor_sent = ''.join(split_strings)

            encoded_emb = self.mention_predictor.predict_json(
                {"anchor_sent": anchor_sent}
            )['contextualized_mention']
            candidate_ent_titles_and_scores = self._candidate_ent_retriever(mention)
            pred_scores = self._cand_ent_candidate_score(mention_vec=encoded_emb,
                                           candidate_ent_titles_and_scores=candidate_ent_titles_and_scores)
            if len(pred_scores) == 0:
                titles, scores = self.kb.search_with_emb(emb=encoded_emb)
                sum_for_normalize = sum(scores)
                pred_scores = [(ent, score / sum_for_normalize) for (ent, score) in zip(titles, scores)]

            predicted_ne.update({'predicted_normalized_entities':
                                     pred_scores
                                 })

        return ne_spans

    def question(self, sentence: str) -> List[Tuple[str, float]]:
        encoded_emb = self.mention_predictor.predict_json(
            {"anchor_sent": sentence}
        )['contextualized_mention']
        titles, scores = self.kb.search_with_emb(emb=encoded_emb)
        sum_score = sum(scores)
        normalized_score = [(entity, score / sum_score) for (entity, score) in zip(titles, scores)]

        return normalized_score

    def _mention2cand_entity_dict_loader(self) -> Dict:
        with open(PRIOR_DICT_PATH, 'r') as f:
            prior_dict = json.load(f)

        return prior_dict

    def _candidate_ent_retriever(self, mention: str,
                                 threshold_prior=0.00001) -> List:
        if mention in self.prior_dict:
            return [(ent, prior) for (ent, prior) in self.prior_dict[mention] if prior >= threshold_prior]
        else:
            return []

    def _cand_ent_candidate_score(self,
                                  mention_vec,
                                  candidate_ent_titles_and_scores) -> List:
        scores = [np.dot(self.kb.title2entity_vec(candidate_ent_titles_and_scores[i][0]), mention_vec) for i in range(len(
            candidate_ent_titles_and_scores
        )) if candidate_ent_titles_and_scores[i][0] in self.kb.entity_title2id]

        if sum(scores) != 0:
            sum_for_normalize = sum(scores)
            scores = [round(score / sum_for_normalize, 4) for score in scores]
        else:
            return candidate_ent_titles_and_scores

        return sorted([(i, j) for (i, j) in zip([ent for (ent, _) in candidate_ent_titles_and_scores], scores)],
                      key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    TXT = '今日は東京都のマックにアップルを買いに行き、スティーブジョブスとドナルドに会い、堀田区に引っ越した。'
    el = EntityLinker()
    q = '日本の総理大臣は？'
    print(el.link(sentence=TXT))
    print(el.question(sentence=q))