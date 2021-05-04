from jel.biencoder.dataset_reader import SmallJaWikiReader
from jel.biencoder.parameters import BiEncoderExperiemntParams
from jel.biencoder.utils import build_vocab, build_data_loaders, build_trainer
from jel.biencoder.encoder import (
    BertPoolerForMention, BertPoolerForTitleAndDef,
    ChiveMentionEncoder, ChiveEntityEncoder
    )
from jel.biencoder.model import Biencoder
from jel.utils.embedder import bert_emb_returner, chive_emb_returner
from typing import Iterable, List, Tuple
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

def biencoder_training() -> Tuple[BasicTextFieldEmbedder, Seq2VecEncoder, Seq2VecEncoder]:
    '''
    :return: embedder, mention_encoder, entity_encoder
    '''
    params = BiEncoderExperiemntParams()
    config = params.opts
    reader = SmallJaWikiReader(config=config)

    # Loading Datasets
    train, dev, test = reader.read('train'), reader.read('dev'), reader.read('test')
    vocab = build_vocab(train)
    vocab.extend_from_instances(dev)

    # TODO: avoid memory consumption and lazy loading
    train, dev, test = list(reader.read('train')), list(reader.read('dev')), list(reader.read('test'))

    train_loader, dev_loader, test_loader = build_data_loaders(config, train, dev, test)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    if config.word_langs_for_training == 'bert':
        embedder = bert_emb_returner()
        mention_encoder, entity_encoder = BertPoolerForMention(word_embedder=embedder), \
                                          BertPoolerForTitleAndDef(word_embedder=embedder)
    elif config.word_langs_for_training == 'chive':
        embedder = chive_emb_returner(vocab=vocab)
        mention_encoder, entity_encoder = ChiveMentionEncoder(word_embedder=embedder), \
                                          ChiveEntityEncoder(word_embedder=embedder)
    else:
        raise NotImplementedError

    model = Biencoder(config, mention_encoder, entity_encoder, vocab)

    trainer = build_trainer(lr=config.lr,
                            num_epochs=config.num_epochs,
                            model=model,
                            train_loader=train_loader,
                            dev_loader=dev_loader)
    trainer.train()

    return embedder, mention_encoder, entity_encoder