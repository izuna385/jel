'''
Seq2VecEncoders for encoding mentions and entities.
'''
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler
from overrides import overrides
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder

class Pooler_for_cano_and_def(Seq2VecEncoder):
    def __init__(self, word_embedding_dropout: float, bert_model_name: str ='japanese_bert',
                 word_embedder: BasicTextFieldEmbedder = BasicTextFieldEmbedder(
                     {'tokens': PretrainedTransformerEmbedder(model_name='cl-tohoku/bert-base-japanese')})):
        super(Pooler_for_cano_and_def, self).__init__()
        self.bert_model_name = bert_model_name
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(word_embedding_dropout)

    def huggingface_nameloader(self):
        if self.bert_name == 'japanese_bert':
            self.bert_weight_filepath = 'cl-tohoku/bert-base-japanese'
        else:
            raise NotImplementedError

    def forward(self, cano_and_def_concatnated_text):
        mask_sent = get_text_field_mask(cano_and_def_concatnated_text)
        entity_emb = self.word_embedder(cano_and_def_concatnated_text)
        entity_emb = self.word_embedding_dropout(entity_emb)
        entity_emb = self.bertpooler_sec2vec(entity_emb, mask_sent)

        return entity_emb


class Pooler_for_mention(Seq2VecEncoder):
    def __init__(self, word_embedding_dropout: float, bert_model_name: str ='japanese_bert',
                 word_embedder: BasicTextFieldEmbedder = BasicTextFieldEmbedder(
                     {'tokens': PretrainedTransformerEmbedder(model_name='cl-tohoku/bert-base-japanese')})):
        super(Pooler_for_mention, self).__init__()
        self.bert_model_name = bert_model_name
        self.huggingface_nameloader()
        self.bertpooler_sec2vec = BertPooler(pretrained_model=self.bert_weight_filepath)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(word_embedding_dropout)

    def huggingface_nameloader(self):
        if self.bert_model_name == 'japanese_bert':
            self.bert_weight_filepath = 'cl-tohoku/bert-base-japanese'
        else:
            raise NotImplementedError

    def forward(self, contextualized_mention):
        mask_sent = get_text_field_mask(contextualized_mention)
        mention_emb = self.word_embedder(contextualized_mention)
        mention_emb = self.word_embedding_dropout(mention_emb)
        mention_emb = self.bertpooler_sec2vec(mention_emb, mask_sent)

        return mention_emb

    @overrides
    def get_output_dim(self):
        # Currently bert-large is not supported.
        return 768