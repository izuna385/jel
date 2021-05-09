from typing import Iterable, List, Tuple
import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader, MultiProcessDataLoader
from allennlp.models import Model
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import Trainer, GradientDescentTrainer
from typing import List, Tuple, Any, Dict, Iterable, Iterator


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
    lr: float,
    num_epochs: int,
    model: Model,
    train_loader: DataLoader,
    dev_loader: DataLoader) -> Trainer:

    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters, lr=lr)
    if torch.cuda.is_available():
        model.cuda()
    trainer = GradientDescentTrainer(
        model=model,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        cuda_device=0 if torch.cuda.is_available() else -1
    )

    return trainer