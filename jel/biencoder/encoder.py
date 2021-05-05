'''
Seq2VecEncoders for encoding mentions and entities.
'''
import torch.nn as nn
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, LstmSeq2VecEncoder, BagOfEmbeddingsEncoder
from allennlp.modules.seq2vec_encoders import BertPooler, CnnEncoder
from overrides import overrides
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
import torch.nn as nn
import torch
import pdb

class BertPoolerForTitleAndDef(Seq2VecEncoder):
    def __init__(self, word_embedding_dropout: float = 0.05, bert_model_name: str = 'japanese_bert',
                 word_embedder: BasicTextFieldEmbedder = BasicTextFieldEmbedder(
                     {'tokens': PretrainedTransformerEmbedder(model_name='cl-tohoku/bert-base-japanese')})):
        super(BertPoolerForTitleAndDef, self).__init__()
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

    def forward(self, cano_and_def_concatnated_text):
        mask_sent = get_text_field_mask(cano_and_def_concatnated_text)
        entity_emb = self.word_embedder(cano_and_def_concatnated_text)
        entity_emb = self.word_embedding_dropout(entity_emb)
        entity_emb = self.bertpooler_sec2vec(entity_emb, mask_sent)

        return entity_emb


class BertPoolerForMention(Seq2VecEncoder):
    def __init__(self, word_embedding_dropout: float = 0.05, bert_model_name: str = 'japanese_bert',
                 word_embedder: BasicTextFieldEmbedder = BasicTextFieldEmbedder(
                     {'tokens': PretrainedTransformerEmbedder(model_name='cl-tohoku/bert-base-japanese')})):
        super(BertPoolerForMention, self).__init__()
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

class ChiveMentionEncoder(Seq2VecEncoder):
    def __init__(self,
                 word_embedder: BasicTextFieldEmbedder,
                 word_embedding_dropout: float = 0.05,):
        super(ChiveMentionEncoder, self).__init__()
        self.sec2vec_for_mention = BagOfEmbeddingsEncoder(embedding_dim=300, averaged=True)
        self.sec2vec_for_context = BagOfEmbeddingsEncoder(embedding_dim=300, averaged=True)
            # LstmSeq2VecEncoder(input_size=300, hidden_size=300, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(600, 300)
        self.linear2 = nn.Linear(300, 300)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(word_embedding_dropout)

    def forward(self, mention, context):
        mask_ment = get_text_field_mask(mention)
        mention_emb = self.word_embedder(mention)
        mention_emb = self.word_embedding_dropout(mention_emb)
        mention_emb = self.sec2vec_for_mention(mention_emb, mask_ment)

        mask_context = get_text_field_mask(context)
        context_emb = self.word_embedder(context)
        context_emb = self.word_embedding_dropout(context_emb)
        context_emb = self.sec2vec_for_context(context_emb, mask_context)

        final_emb = self.linear(torch.cat((mention_emb, context_emb), 1))
        final_emb = self.linear2(final_emb)

        return final_emb


class ChiveEntityEncoder(Seq2VecEncoder):
    def __init__(self,
                 word_embedder: BasicTextFieldEmbedder,
                 word_embedding_dropout: float = 0.05):
        super(ChiveEntityEncoder, self).__init__()
        self.sec2vec_for_title = BagOfEmbeddingsEncoder(embedding_dim=300, averaged=True)
        self.sec2vec_for_ent_desc = BagOfEmbeddingsEncoder(embedding_dim=300, averaged=True)
            # LstmSeq2VecEncoder(input_size=300, hidden_size=300, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(600, 300)
        self.linear2 = nn.Linear(300, 300)
        self.word_embedder = word_embedder
        self.word_embedding_dropout = nn.Dropout(word_embedding_dropout)

    def forward(self, title, ent_desc):
        mask_title = get_text_field_mask(title)
        title_emb = self.word_embedder(title)
        title_emb = self.word_embedding_dropout(title_emb)
        title_emb = self.sec2vec_for_title(title_emb, mask_title)

        mask_ent_desc = get_text_field_mask(ent_desc)
        ent_desc_emb = self.word_embedder(ent_desc)
        ent_desc_emb = self.word_embedding_dropout(ent_desc_emb)
        ent_desc_emb = self.sec2vec_for_ent_desc(ent_desc_emb, mask_ent_desc)

        final_emb = self.linear(torch.cat((title_emb, ent_desc_emb), 1))
        final_emb = self.linear2(final_emb)

        return final_emb