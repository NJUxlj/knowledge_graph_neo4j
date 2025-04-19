# -*- coding: utf-8 -*-
import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data
from sklearn.metrics import classification_report
"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        
        self.index_to_label = None
        
        
        
    def eval(self, epoch):
        self.logger.info("开始测试第i%d轮模型的效果" % epoch)
        self.stats_dict = {"object_acc":0, "attribute_acc":0, "value_acc":0, "full_match_acc":0}  # 含义： 实体1识别准确率， 关系识别准确率， 实体2识别准确率， 三者完全匹配准确率