'''
ja-wiki dataset reader for training bi-encoder
'''
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import SpanField, ListField, TextField, MetadataField, ArrayField, SequenceLabelField, LabelField
from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from typing import List, Tuple, Any, Dict, Iterable
import random
import pdb
from tqdm import tqdm
from jel.utils.common import jopen
from jel.utils.tokenizer import JapaneseBertTokenizer
import numpy as np

from jel.common_config import (
    MENTION_ANCHORS,
    MENTION_START_BERT_TOKEN, MENTION_END_BERT_TOKEN,
    CANONICAL_AND_DEF_BERT_CONNECT_TOKEN,
    CLS_TOKEN, SEP_TOKEN,
    MENTION_ANCHORS_REGEX,
    MENTION_START_ANCHOR, MENTION_END_ANCHOR
)


class SmallJaWikiReader(DatasetReader):
    def __init__(
        self,
        config,
        resource_save_dir='./',
        eval=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = JapaneseBertTokenizer(resource_save_dir=resource_save_dir)
        self.token_indexers = self.tokenizer.token_indexer_returner()
        self.config = config
        self.eval = eval

        # kb_load
        self.id2title, self.title2id, self.id2ent_doc = self._kb_loader()

    def _train_loader(self) -> dict:
        data = jopen(file_path=self.config.biencoder_dataset_file_path)
        return data['train']

    def _dev_loader(self) -> dict:
        data = jopen(file_path=self.config.biencoder_dataset_file_path)

        return data['dev']

    def _test_loader(self) -> dict:
        data = jopen(file_path=self.config.biencoder_dataset_file_path)

        return data['test']

    def _title2doc_loader(self) -> dict:
        return jopen(file_path=self.config.title2doc_file_path)

    def _kb_loader(self) -> Tuple[Dict[int, Any], Dict[Any, int], Dict[int, Any]]:
        title2ent_doc = self._title2doc_loader()

        id2title, title2id, id2ent_doc = {}, {}, {}

        for title, ent_doc in title2ent_doc.items():
            assert len(id2title) == len(title2id)
            assert len(id2title) == len(id2ent_doc)
            idx = len(id2title)
            if title not in title2id:
                id2title.update({idx: title})
                title2id.update({title: idx})
                id2ent_doc.update({idx: ent_doc})

        return id2title, title2id, id2ent_doc

    @overrides
    def _read(self, train_dev_test_flag: str) -> Iterable[Instance]:
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

        for data in tqdm(enumerate(dataset)):
            data = self._one_line_parser(data=data, train_dev_test_flag=train_dev_test_flag)
            try:
                yield self.text_to_instance(data)

            except:
                print("parseError", data["anchor_sent"])
                continue

            #     TODO: print parseError
            #     continue


    def _one_line_parser(self, data, train_dev_test_flag='train') -> dict:
        mention_idx, mention_data = int(data[0]), data[1]

        document_title = mention_data['document_title']
        anchor_sent = mention_data['anchor_sent']
        annotation_doc_entity_title = mention_data['annotation_doc_entity_title']
        mention = mention_data['mention']
        original_sentence = mention_data['original_sentence']
        original_sentence_mention_start = mention_data['original_sentence_mention_start']
        original_sentence_mention_end = mention_data['original_sentence_mention_end']

        tokenized_context_including_target_anchors = self.tokenizer.tokenize(txt=anchor_sent)
        tokenized_context_including_target_anchors = self._mention_split_tokens_converter(tokenized_context_including_target_anchors)
        data = {'context': tokenized_context_including_target_anchors}

        if annotation_doc_entity_title in self.title2id:
            data['gold_ent_idx'] = self.title2id[annotation_doc_entity_title]
        else:
            data['gold_ent_idx'] = -1

        data['gold_title_and_def'] = self._title_and_ent_doc_concatenator(title=annotation_doc_entity_title)

        return data

    def _mention_split_tokens_converter(self, tokens: List[str]) -> List[Token]:
        '''

        :param tokens:
        :return: Tokens after considering window size
        '''
        left, mention, right = list(), list(), list()
        assert MENTION_START_ANCHOR in tokens
        assert MENTION_END_ANCHOR in tokens

        l_flag, m_flag = 0, 0
        for str_tok in tokens:
            if str_tok in MENTION_START_BERT_TOKEN:
                l_flag += 1
                continue
            if str_tok in MENTION_END_BERT_TOKEN:
                m_flag += 1
                continue

            if l_flag == 0 and m_flag == 0:
                left.append(str_tok)

            if l_flag == 1 and m_flag == 0:
                mention.append(str_tok)

            if l_flag == 1 and m_flag == 1:
                right.append(str_tok)

        left = left[-self.config.max_context_window_size:]
        mention = mention[:self.config.max_mention_size]
        right = right[:self.config.max_context_window_size]

        window_condidered_tokens = list()
        window_condidered_tokens.append(CLS_TOKEN)
        window_condidered_tokens += left
        window_condidered_tokens.append(MENTION_START_BERT_TOKEN)
        window_condidered_tokens += mention
        window_condidered_tokens.append(MENTION_END_BERT_TOKEN)
        window_condidered_tokens += right
        window_condidered_tokens.append(SEP_TOKEN)

        return [Token(tok) for tok in window_condidered_tokens]

    def _title_and_ent_doc_concatenator(self, title: str) -> List[Token]:
        tokenized_title =  self.tokenizer.tokenize(txt=title)[:self.config.max_title_token_size]

        ent_doc_sentences = ''.join(self.id2ent_doc[self.title2id[title]][:self.config.max_ent_considered_sent_num])
        tokenized_ent_desc_tokens = self.tokenizer.tokenize(txt=ent_doc_sentences)[:self.config.max_ent_desc_token_size]

        concatenated_tokens = list()
        concatenated_tokens.append(CLS_TOKEN)
        concatenated_tokens += tokenized_title
        concatenated_tokens.append(CANONICAL_AND_DEF_BERT_CONNECT_TOKEN)
        concatenated_tokens += tokenized_ent_desc_tokens
        concatenated_tokens.append(SEP_TOKEN)

        return [Token(tok) for tok in concatenated_tokens]

    @overrides
    def text_to_instance(self, data=None) -> Instance:
        context_field = TextField(data['context'], self.token_indexers)
        fields = {"context": context_field}

        fields['gold_ent_idx'] = ArrayField(np.array(data['gold_ent_idx']))
        fields['gold_title_and_def'] = TextField(data['gold_title_and_def'], self.token_indexers)

        return Instance(fields)