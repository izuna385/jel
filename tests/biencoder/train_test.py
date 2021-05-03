from jel.biencoder.train import biencoder_training
from allennlp.models import Model

def train_test():
    model = biencoder_training()
    # assert type(model) == Model