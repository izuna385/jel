import transformers
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer, PretrainedTransformerIndexer
import os
import urllib.request
from typing import List, Tuple
import pdb

class JapaneseBertTokenizer:
    def __init__(self, bert_model_name: str ='japanese_bert',
                 resource_save_dir: str = './'
    ):
        self.bert_model_name = bert_model_name
        self.resource_save_dir = resource_save_dir

        # load tokenizer
        self._bert_model_and_vocab_downloader()
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
            return transformers.BertTokenizer(vocab_file=vocab_file,
                                              do_basic_tokenize=True)
        else:
            raise NotImplementedError('Currently {} are not supported.'.format(self.bert_model_name))

    def tokenize(self, txt: str, remove_special_vocab=False) -> List[str]:
        if remove_special_vocab:
            split_to_subwords = self.bert_tokenizer.tokenize(txt)
            new_tokens = list()

            for token in split_to_subwords:
                if token in ['[CLS]', '[SEP]']:
                    continue
                new_tokens.append(token)

            return new_tokens
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