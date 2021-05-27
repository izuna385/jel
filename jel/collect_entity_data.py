from jel.utils.common import jopen
from glob import glob
from jel.biencoder.predictor import predictors_loader
from tqdm import tqdm
import os
from jel.common_config import ENTITY_DATA_PATH, ENTITY_VEC_DIR_PATH
import logging
import pickle
logger = logging.getLogger(__name__)


class EntityCollector:
    def __init__(self,
                 max_token_in_one_entity_name: int =10,
                 max_token_in_one_sentence_of_entity_desc: int = 100,
                 max_sent_from_one_entity: int = 3,
                 debug: bool = False):
        self.json_paths = glob(ENTITY_DATA_PATH+'.json')
        print('all jsons:', len(self.json_paths))
        if debug:
            self.json_paths = self.json_paths[:1000]
        self.max_token_in_one_entity_name = max_token_in_one_entity_name
        self.max_token_in_one_sentence_of_entity_desc = max_token_in_one_sentence_of_entity_desc
        self.max_sent_from_one_entity = max_sent_from_one_entity
        _, self.entity_encoder = predictors_loader()

        if not os.path.exists(ENTITY_VEC_DIR_PATH):
            os.makedirs(ENTITY_VEC_DIR_PATH)

    def _from_json_entity_data_returner(self, json_path: str):
        entity_data = jopen(json_path)['doc_title2sents']
        entity_names, descriptions = list(), list()

        for entity_name, tokenized_data in entity_data.items():
            tokenized_title = tokenized_data['sudachi_tokenized_title'][:self.max_token_in_one_entity_name]
            tokenized_descs = tokenized_data['sudachi_tokenized_sents'][:self.max_sent_from_one_entity]
            tokenized_descs = [token for token in [tokenized_sent[:self.max_token_in_one_sentence_of_entity_desc] for
                               tokenized_sent in tokenized_descs]]
            tokenized_descs = [item for sublist in tokenized_descs for item in sublist]
            entity_names.append(tokenized_title)
            descriptions.append(tokenized_descs)

        assert len(entity_names) == len(descriptions)

        return entity_names, descriptions

    def _entity_data_loader(self):
        for json_path in self.json_paths:
            yield self._from_json_entity_data_returner(json_path=json_path)

    def _batched_entity_name2vec_dumper(self, unique_idx, batched_entity_names, vecs):
        with open(ENTITY_VEC_DIR_PATH + str(unique_idx)+'.pkl', 'wb') as f:
            pickle.dump([{"entity_name": entity_name, "vec": vec} for
                         (entity_name, vec) in zip(batched_entity_names, vecs)],
                        f)

    def entity2vec_creator(self):
        '''
        create entity2vec file from preprocessed sudachi wiki.
        :return:
        '''
        logger.debug(msg='iterate over {} jsons'.format(len(self.json_paths)))
        print('iterate over {} jsons'.format(len(self.json_paths)))
        for idx, (batched_entity_names, batched_entity_descriptions) in tqdm(enumerate(self._entity_data_loader())):
            batched_dict = list({"gold_title": title, "gold_ent_desc": desc} for (title, desc) in
                                zip(batched_entity_names, batched_entity_descriptions))
            batched_entity_vecs = self.entity_encoder.predict_batch_json(batched_dict)
            self._batched_entity_name2vec_dumper(unique_idx=idx,
                                                 batched_entity_names=batched_entity_names,
                                                 vecs=batched_entity_vecs)


if __name__ == '__main__':
    entity_collector = EntityCollector()
    entity_collector.entity2vec_creator()

