# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


'''
复现论文Relation Bert
建立网络模型结构
'''
class RBert(nn.Module):
    def __init__(self, config):
        super(RBert, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.bert_path)