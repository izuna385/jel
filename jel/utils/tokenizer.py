import transformers
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
import os
import urllib.request
from typing import List, Tuple
import pdb
import re
from allennlp.data.token_indexers import SingleIdTokenIndexer
from sudachipy import tokenizer as sudachiTokenizer
from sudachipy import dictionary as sudachiDic

from jel.common_config import (
    MENTION_ANCHORS,
    MENTION_START_BERT_TOKEN, MENTION_END_BERT_TOKEN,
    CANONICAL_AND_DEF_BERT_CONNECT_TOKEN,
    CLS_TOKEN, SEP_TOKEN,
    MENTION_ANCHORS_REGEX,
    MENTION_START_ANCHOR, MENTION_END_ANCHOR
)


class SudachiTokenizer:
    def __init__(self,
                 mention_anchors: Tuple[str] = MENTION_ANCHORS
    ):
        '''
        :param resource_save_dir:
        :param mention_anchors:
        '''
        self.tokenizer = sudachiDic.Dictionary().create()
        self.mode = sudachiTokenizer.Tokenizer.SplitMode.B
        self.mention_anchors = mention_anchors

    def tokenize(self, txt: str) -> List[str]:
        # First, check whether text contains mention anchors.
        mention_anchor_exist_flag = 0
        for anchor in self.mention_anchors:
            if anchor in txt:
                mention_anchor_exist_flag += 1
                break

        if mention_anchor_exist_flag:
            texts = re.split(MENTION_ANCHORS_REGEX, txt)
            try:
                assert len(texts) == 3
            except:
                print("bad tokenize: {}".format(txt))
                texts = texts[:3]
            tokens = list()
            tokens += [m.surface() for m in self.tokenizer.tokenize(texts[0], self.mode)]
            tokens.append(MENTION_START_ANCHOR)
            tokens += [m.surface() for m in self.tokenizer.tokenize(texts[1], self.mode)]
            tokens.append(MENTION_END_ANCHOR)
            tokens += [m.surface() for m in self.tokenizer.tokenize(texts[2], self.mode)]

            return tokens
        else:
            return [m.surface() for m in self.tokenizer.tokenize(txt, self.mode)]

    def token_indexer_returner(self):
        return {"tokens": SingleIdTokenIndexer()}


class JapaneseBertTokenizer:
    def __init__(self, bert_model_name: str ='japanese_bert',
                 resource_save_dir: str = './',
                 mention_anchors: Tuple[str] = MENTION_ANCHORS
    ):
        '''
        :param bert_model_name:
        :param resource_save_dir:
        :param special_anchors:
        '''

        self.bert_model_name = bert_model_name
        self.resource_save_dir = resource_save_dir
        self.mention_anchors = mention_anchors
        assert len(self.mention_anchors) == 2

        # load tokenizer
        # self._bert_model_and_vocab_downloader()
        self.bert_tokenizer = self.bert_tokenizer_returner()

    def _huggingfacename_returner(self) -> Tuple:
        'Return huggingface modelname and do_lower_case parameter'
        if self.bert_model_name == 'japanese_bert':
            return 'cl-tohoku/bert-base-japanese', False
        else:
            raise NotImplementedError('Currently {} are not supported.'.format(self.bert_model_name))

    def token_indexer_returner(self) -> dict:
        huggingface_name, do_lower_case = self._huggingfacename_returner()
        return {'tokens': PretrainedTransformerIndexer(
            model_name=huggingface_name,
            # do_lowercase=do_lower_case
        )
        }

    def bert_tokenizer_returner(self):
        if self.bert_model_name == 'japanese_bert':
            vocab_file = self.resource_save_dir + 'vocab_file/vocab.txt'
            # return transformers.BertTokenizer(vocab_file=vocab_file,
            #                                   do_basic_tokenize=True,
            #                                   never_split=list(set(MENTION_ANCHORS)))
            return transformers.BertTokenizer.from_pretrained(
                pretrained_model_name_or_path='cl-tohoku/bert-base-japanese',
                never_split=list(set(MENTION_ANCHORS))
            )
        else:
            raise NotImplementedError('Currently {} are not supported.'.format(self.bert_model_name))

    def tokenize(self, txt: str, remove_special_vocab=False) -> List[str]:
        # First, check whether text contains mention anchors.
        mention_anchor_exist_flag = 0
        for anchor in self.mention_anchors:
            if anchor in txt:
                mention_anchor_exist_flag += 1
                break

        if remove_special_vocab:
            split_to_subwords = self.bert_tokenizer.tokenize(txt)
            new_tokens = list()

            for token in split_to_subwords:
                if token in ['[CLS]', '[SEP]']:
                    continue

                new_tokens.append(token)

            return new_tokens
        else:
            if mention_anchor_exist_flag:
                texts = re.split(MENTION_ANCHORS_REGEX, txt)
                try:
                    assert len(texts) == 3
                except:
                    print("bad tokenize: {}".format(txt))
                    texts = texts[:3]
                tokens = list()
                tokens += self.bert_tokenizer.tokenize(texts[0])
                tokens.append(MENTION_START_ANCHOR)
                tokens += self.bert_tokenizer.tokenize(texts[1])
                tokens.append(MENTION_END_ANCHOR)
                tokens += self.bert_tokenizer.tokenize(texts[2])

                return tokens
            else:

                return self.bert_tokenizer.tokenize(txt)

    def _bert_model_and_vocab_downloader(self) -> None:
        resource_saved_dict = self.resource_save_dir + self.bert_model_name + '/'

        if not os.path.exists(resource_saved_dict):
            os.mkdir(resource_saved_dict)
            print('=== Downloading japanese-bert ===')
            # https://huggingface.co/cl-tohoku/bert-base-japanese
            urllib.request.urlretrieve("https://huggingface.co/cl-tohoku/bert-base-japanese/raw/main/config.json", resource_saved_dict + 'config.json')
            urllib.request.urlretrieve("https://huggingface.co/cl-tohoku/bert-base-japanese/raw/main/pytorch_model.bin", resource_saved_dict + 'pytorch_model.bin')
            urllib.request.urlretrieve("https://huggingface.co/cl-tohoku/bert-base-japanese/raw/main/tokenizer_config.json", resource_saved_dict + 'tokenizer_config.json')

        if not os.path.exists(resource_saved_dict+'vocab_file/'):
            os.mkdir(resource_saved_dict+'./vocab_file/')
            urllib.request.urlretrieve("https://huggingface.co/cl-tohoku/bert-base-japanese/raw/main/vocab.txt", './vocab_file/vocab.txt')