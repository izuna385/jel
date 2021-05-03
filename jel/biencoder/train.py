from jel.biencoder.dataset_reader import SmallJaWikiReader
from jel.biencoder.parameters import BiEncoderExperiemntParams
from jel.biencoder.utils import build_vocab, build_data_loaders, emb_returner, build_trainer
from jel.biencoder.encoder import Pooler_for_mention, Pooler_for_cano_and_def
from jel.biencoder.model import Biencoder
from allennlp.training.util import evaluate
import copy

def biencoder_training():
    params = BiEncoderExperiemntParams()
    config = params.opts
    reader = SmallJaWikiReader(config=config)

    # Loading Datasets
    train, dev, test = reader.read('train'), reader.read('dev'), reader.read('test')
    vocab = build_vocab(train)
    vocab.extend_from_instances(dev)

    train, dev, test = list(reader.read('train')), list(reader.read('dev')), list(reader.read('test'))
    train_loader, dev_loader, test_loader = build_data_loaders(config, train, dev, test)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    embedder = emb_returner()
    mention_encoder, entity_encoder = Pooler_for_mention(word_embedder=embedder), \
                                      Pooler_for_cano_and_def(word_embedder=embedder)

    model = Biencoder(mention_encoder, entity_encoder, vocab)

    trainer = build_trainer(lr=config.lr,
                            num_epochs=config.num_epochs,
                            model=model,
                            train_loader=train_loader,
                            dev_loader=dev_loader)
    trainer.train()

    return model