__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'



import torch
import torch.nn as nn

from models.modeling_bert import BertForPostTraining, Config
from torch.nn import CrossEntropyLoss, Dropout, Embedding, Softmax


class BertForNLIFineTune(nn.Module):
    """
    https://github.com/kakaobrain/KorNLUDatasets
    NLI Fine-Tuning 을 위한 nn.Module 서브 클래스
    """
    def __init__(self, config: Config) -> None:
        super(BertForNLIFineTune, self).__init__()
        self.config = config
        self.model = BertForPostTraining(self.config)

    def forward(self, batch: dict):
        pass