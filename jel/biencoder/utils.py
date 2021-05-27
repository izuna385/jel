import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
)
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.models import Model
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from typing import List, Tuple, Any, Dict, Iterable, Iterator
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, LstmSeq2VecEncoder, BagOfEmbeddingsEncoder
import logging
import os
import shutil


logger = logging.getLogger(__name__)


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_data_loaders(config,
    dataset_reader: DatasetReader) -> Tuple[MultiProcessDataLoader, MultiProcessDataLoader, MultiProcessDataLoader]:

    train_loader = MultiProcessDataLoader(dataset_reader, data_path='train', batch_size=config.batch_size_for_train, shuffle=False)
    dev_loader = MultiProcessDataLoader(dataset_reader, data_path='dev', batch_size=config.batch_size_for_eval, shuffle=False)
    test_loader = MultiProcessDataLoader(dataset_reader, data_path='test', batch_size=config.batch_size_for_eval, shuffle=False)

    return train_loader, dev_loader, test_loader

def build_trainer(
    config,
    lr: float,
    serialization_dir: str,
    num_epochs: int,
    model: Model,
    train_loader: DataLoader,
    dev_loader: DataLoader) -> Trainer:

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=lr)
    if torch.cuda.is_available():
        model.cuda()

    # remove serialization dir
    if os.path.exists(serialization_dir) and config.shutil_pre_finished_experiment:
        shutil.rmtree(serialization_dir)

    if not os.path.exists(serialization_dir):
        os.makedirs(serialization_dir)

    trainer = GradientDescentTrainer(
        model=model,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        serialization_dir=serialization_dir,
        cuda_device=0 if torch.cuda.is_available() else -1
    )

    return trainer

def encoder_saver(encoder:Seq2VecEncoder,
                  path: str) -> None:
    torch.save(encoder.state_dict(), path)

def encoder_loader(encoder: Seq2VecEncoder,
                   path: str) -> Seq2VecEncoder:
    encoder.load_state_dict(torch.load(path))

    return encoder

def vocab_loader(vocab_dir_path: str) -> Vocabulary:
    vocab = Vocabulary.from_files(directory=vocab_dir_path)

    return vocab