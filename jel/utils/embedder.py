from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import PretrainedTransformerEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from jel.common_config import CACHE_ROOT
from allennlp.modules.token_embedders.embedding import _read_embeddings_from_text_file

def bert_emb_returner():
    return BasicTextFieldEmbedder(
                     {'tokens': PretrainedTransformerEmbedder(model_name='cl-tohoku/bert-base-japanese')})

def chive_emb_returner(vocab: Vocabulary) -> BasicTextFieldEmbedder:
    # embed_matrix = _read_embeddings_from_text_file(
    #     file_uri="./resources/chive-1.1-mc30.txt",
    #     embedding_dim=300,
    #     vocab=vocab
    # )

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=300,
                                pretrained_file=str(CACHE_ROOT)+"/resources/chive-1.1-mc30.txt",
                                vocab=vocab)

    return BasicTextFieldEmbedder({'tokens': token_embedding})