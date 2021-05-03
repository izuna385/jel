
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from typing import List

import os
import random
import pdb
from tqdm import tqdm
import json
from jel.utils.common import jopen
from jel.utils.tokenizer import JapaneseBertTokenizer
import numpy as np

class SmallJaWikiReader(DatasetReader):
    def __init__(
        self,
        config,
        resource_save_dir='./',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = JapaneseBertTokenizer(resource_save_dir=resource_save_dir)
        self.token_indexers = self.tokenizer.token_indexer_returner()
        self.config = config

    def _train_loader(self) -> dict:
        data = jopen(file_path=self.config.biencoder_dataset_file_path)
        return data['train']

    def _dev_loader(self) -> dict:
        data = jopen(file_path=self.config.biencoder_dataset_file_path)

        return data['dev']

    def _test_loader(self) -> dict:
        data = jopen(file_path=self.config.biencoder_dataset_file_path)

        return data['test']

    def _title2doc_loader(self):
        return jopen(file_path=self.config.title2doc_file_path)

    @overrides
    def _read(self, train_dev_test_flag: str) -> List:
        '''
        :param train_dev_test_flag: 'train', 'dev', 'test'
        :return: list of instances
        '''
        if train_dev_test_flag == 'train':
            dataset = self._train_loader()
            random.shuffle(dataset)
        elif train_dev_test_flag == 'dev':
            dataset = self._dev_loader()
        elif train_dev_test_flag == 'test':
            dataset = self._test_loader()
        else:
            raise NotImplementedError(
                "{} is not a valid flag. Choose from train, dev and test".format(train_dev_test_flag))

        if self.config.debug:
            dataset = dataset[:self.config.debug_data_num]

        ignored_mentions_num = 0

        all_parsed_data = list()

        for data in tqdm(enumerate(dataset)):
            # TODO: yield
            parsed_data = self._one_line_parser(data=data, train_dev_test_flag=train_dev_test_flag)
            all_parsed_data.append(parsed_data)

            # except:
            #     TODO: print parseError
            #     continue


    def _one_line_parser(self, data, train_dev_test_flag='train'):
        mention_idx, mention_data = int(data[0]), data[1]

        document_title = mention_data['document_title']
        anchor_sent = mention_data['anchor_sent']
        annotation_doc_entity_title = mention_data['annotation_doc_entity_title']
        mention = mention_data['mention']
        original_sentence = mention_data['original_sentence']
        original_sentence_mention_start = mention_data['original_sentence_mention_start']
        original_sentence_mention_end = mention_data['original_sentence_mention_end']

        if train_dev_test_flag in ['train'] or (train_dev_test_flag == 'dev' and self.dev_eval_flag == 0):
            line = self.id2mention[mention_uniq_id]
            gold_dui, _, gold_surface_mention, target_anchor_included_sentence = line.split('\t')
            tokenized_context_including_target_anchors = self.custom_tokenizer_class.tokenize(
                txt=target_anchor_included_sentence)
            tokenized_context_including_target_anchors = [Token(split_token) for split_token in
                                                          tokenized_context_including_target_anchors]
            data = {'context': tokenized_context_including_target_anchors}

            data['mention_uniq_id'] = int(mention_uniq_id)
            data['gold_duidx'] = int(self.dui2idx[gold_dui]) if gold_dui in self.dui2idx and gold_dui in self.dui2canonical else -1
            if gold_dui in self.dui2canonical:
                data['gold_dui_canonical_and_def_concatenated'] = self._canonical_and_def_context_concatenator(dui=gold_dui)
        else:
            assert train_dev_test_flag in ['dev', 'test']
            line = self.id2mention[mention_uniq_id]
            gold_dui, _, surface_mention, target_anchor_included_sentence = line.split('\t')

            candidate_duis_idx = [self.dui2idx[dui] for dui in self.candidate_generator.mention2candidate_duis[surface_mention]
                              if dui in self.dui2idx and dui in self.dui2canonical][:self.config.max_candidates_num]
            while len(candidate_duis_idx) < self.config.max_candidates_num:
                random_choiced_dui = random.choice([dui for dui in self.dui2idx.keys()])
                if self.dui2idx[random_choiced_dui] not in candidate_duis_idx:
                    candidate_duis_idx.append(self.dui2idx[random_choiced_dui])
            tokenized_context_including_target_anchors = self.custom_tokenizer_class.tokenize(
                txt=target_anchor_included_sentence)
            tokenized_context_including_target_anchors = [Token(split_token) for split_token in
                                                          tokenized_context_including_target_anchors][:self.config.max_context_len]
            data = {'context': tokenized_context_including_target_anchors}
            data['candidate_duis_idx'] = candidate_duis_idx
            data['gold_duidx'] = int(self.dui2idx[gold_dui]) if gold_dui in self.dui2idx else -1

            gold_location_in_candidates = [0 for _ in range(self.config.max_candidates_num)]

            if gold_dui in self.dui2idx:
                for idx, cand_idx in enumerate(candidate_duis_idx):
                    if cand_idx == self.dui2idx[gold_dui]:
                        gold_location_in_candidates[idx] += 1

                        if train_dev_test_flag == 'dev':
                            self.dev_recall += 1
                        if train_dev_test_flag == 'test':
                            self.test_recall += 1

            data['gold_location_in_candidates'] = gold_location_in_candidates
            data['mention_uniq_id'] = int(mention_uniq_id)

        return data

    def _canonical_and_def_context_concatenator(self, dui):
        canonical =  self.custom_tokenizer_class.tokenize(txt=self.dui2canonical[dui])
        definition =  self.custom_tokenizer_class.tokenize(txt=self.dui2definition[dui])
        concatenated = ['[CLS]']
        concatenated += canonical[:self.config.max_canonical_len]
        concatenated.append(CANONICAL_AND_DEF_CONNECTTOKEN)
        concatenated += definition[:self.config.max_def_len]
        concatenated.append('[SEP]')

        return [Token(tokenized_word) for tokenized_word in concatenated]

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        context_field = TextField(data['context'], self.token_indexers)
        fields = {"context": context_field}

        fields['gold_duidx'] = ArrayField(np.array(data['gold_duidx']))
        fields['mention_uniq_id'] = ArrayField(np.array(data['mention_uniq_id']))

        if data['mention_uniq_id'] in self.test_mention_ids or \
                (data['mention_uniq_id'] in self.dev_mention_ids and self.dev_eval_flag):
            candidates_canonical_and_def_concatenated = [TextField(self._canonical_and_def_context_concatenator(
                dui=self.idx2dui[idx]), self.token_indexers) for idx in data['candidate_duis_idx']]
            fields['candidates_canonical_and_def_concatenated'] = ListField(candidates_canonical_and_def_concatenated)
            fields['gold_location_in_candidates'] = ArrayField(np.array([data['gold_location_in_candidates']],
                                                                        dtype='int16'))
            fields['gold_dui_canonical_and_def_concatenated'] = MetadataField(0)
        else: # train, or dev-eval under train
            fields['candidates_canonical_and_def_concatenated'] = MetadataField(0)
            fields['gold_location_in_candidates'] = MetadataField(0)
            fields['gold_dui_canonical_and_def_concatenated'] = TextField(
                data['gold_dui_canonical_and_def_concatenated'],
                self.token_indexers)

        return Instance(fields)