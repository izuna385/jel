from jel.biencoder.dataset_reader import SmallJaWikiReader
from jel.biencoder.parameters import BiEncoderExperiemntParams
from jel.biencoder.utils import build_vocab, build_data_loaders, build_trainer, encoder_saver
from jel.biencoder.encoder import (
    BertPoolerForMention, BertPoolerForTitleAndDef,
    ChiveMentionEncoder, ChiveEntityEncoder
    )
from jel.biencoder.model import Biencoder
from jel.utils.embedder import bert_emb_returner, chive_emb_returner
from typing import Iterable, List, Tuple
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
import logging
import pdb
import os
import shutil

from jel.common_config import ENCODER_DIRPATH

logger = logging.getLogger(__name__)

def biencoder_train_and_save_params(debug=False) -> Tuple[BasicTextFieldEmbedder, Seq2VecEncoder, Seq2VecEncoder]:
    '''
    :return: embedder, mention_encoder, entity_encoder
    '''
    params = BiEncoderExperiemntParams()
    config = params.opts
    if debug:
        config.debug = True

    reader = SmallJaWikiReader(config=config)
    reader._kb_loader()
    # Loading Datasets
    train, dev, test = reader.read('train'), reader.read('dev'), reader.read('test')
    vocab = build_vocab(train)
    vocab.extend_from_instances(dev), vocab.extend_from_instances(test)
    try:
        shutil.rmtree(config.vocab_dir)
    except:
        pass
    try:
        os.makedirs(config.vocab_dir)
    except:
        pass

    vocab.save_to_files(config.vocab_dir)

    train_loader, dev_loader, test_loader = build_data_loaders(config, reader)
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

    trainer = build_trainer(config=config,
                            lr=config.lr,
                            serialization_dir=config.serialization_dir,
                            num_epochs=config.num_epochs,
                            model=model,
                            train_loader=train_loader,
                            dev_loader=dev_loader)
    trainer.train()

    logger.debug(msg='saving mention and entity encoder')

    if not os.path.exists(ENCODER_DIRPATH):
        os.makedirs(ENCODER_DIRPATH)

    encoder_saver(mention_encoder, os.path.join(ENCODER_DIRPATH, 'mention_encoder.th'))
    encoder_saver(entity_encoder, os.path.join(ENCODER_DIRPATH, 'entity_encoder.th'))

    return embedder, mention_encoder, entity_encoder