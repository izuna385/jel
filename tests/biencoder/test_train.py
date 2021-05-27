from jel.biencoder.train import biencoder_train_and_save_params
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
import pdb
#
# def test_biencoder_training():
#     embedder, mention_encoder, entity_encoder = biencoder_train_and_save_params(debug=False)