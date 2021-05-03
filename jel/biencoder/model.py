'''
Model classes
'''
import torch
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model
from overrides import overrides
from allennlp.training.metrics import CategoricalAccuracy, BooleanAccuracy
from torch.nn.functional import normalize
import torch.nn.functional as F
import torch.nn as nn
import pdb

class Biencoder(Model):
    def __init__(self,
                 mention_encoder: Seq2VecEncoder,
                 entity_encoder: Seq2VecEncoder,
                 vocab,
                 scoring_function_for_model: str='cossim'):

        super().__init__(vocab)
        self.scoring_function_for_model = scoring_function_for_model
        self.mention_encoder = mention_encoder
        self.accuracy = CategoricalAccuracy()
        self.entity_encoder = entity_encoder

        self.istrainflag = 1 # Immutable

    def forward(self,
                context: torch.Tensor = None,
                gold_ent_idx: torch.Tensor = None,
                gold_title_and_def: torch.Tensor = None
                ):

        batch_num = context['tokens']['token_ids'].size(0)
        device = torch.get_device(context['tokens']['token_ids']) if torch.cuda.is_available() else torch.device('cpu')
        contextualized_mention = self.mention_encoder(context)
        encoded_entites = self.entity_encoder(cano_and_def_concatnated_text=gold_title_and_def)

        if self.scoring_function_for_model == 'cossim':
            contextualized_mention = normalize(contextualized_mention, dim=1)
            encoded_entites = normalize(encoded_entites, dim=1)

        encoded_entites = encoded_entites.squeeze(1)
        dot_product = torch.matmul(contextualized_mention, encoded_entites.t())  # [bs, bs]
        mask = torch.eye(batch_num).to(device)
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()

        output = {'loss': loss}

        if self.istrainflag:
            golds = torch.eye(batch_num).to(device)
            self.accuracy(dot_product, torch.argmax(golds, dim=1))

        else:
            output['gold_duidx'] = gold_ent_idx
            output['encoded_mentions'] = contextualized_mention

        return output

    @overrides
    def get_metrics(self, reset: bool = False):
        return {"accuracy": self.accuracy.get_metric(reset)}

    def return_entity_encoder(self):
        return self.entity_encoder