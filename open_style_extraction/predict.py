# -*- coding: utf-8 -*-
import torch
import re
import json
import numpy as np
from collections import defaultdict
from config import Config
from model import TorchModel

"""
模型效果测试
"""

class SentenceLabel:
    def __init__(self, config, model_path):
        self.config = config
        self.schema = {"B_object":0,
                       "I_object":1,
                       "B_attribute":2,
                       "I_attribute":3,
                       "B_value":4,
                       "I_value":5,
                       "O":6}
        
        self.vocab = self.load_vocab(config["vocab_path"])
        
        self.model = TorchModel(config)
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        
        print("模型加载完毕~~~")
        
        
    def load_vocab(self, vocab_path):
        vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                vocab[line] = len(vocab) + 1
                
        self.config["vocab_size"] = len(vocab)
        return vocab
        
    def seek_pattern(self, pattern, pred_label, context):
        '''
        pattern: 正则表达式模板
        pred_label: shape = (sen_len, )
        context: str: 待标注的句子
        '''
        pred_obj:re.Match = re.search(pattern, pred_label)
        
        if pred_obj:
            start,end = pred_obj.span()
            pred_obj = context[start:end]
            
        else:
            pred_obj  = ""
        return pred_obj
    
    
    
    def decode(self, pred_label, context):
        '''
        pred_label: shape = (sen_len, )
        
        context: str: 待标注的句子
        '''
        pred_label = "".join([str(i) for i in pred_label.detach().tolist()])
        
        # 匹配以0开头，后接多个1的标签模式（对应B-object + I-object*）
        pred_obj = self.seek_pattern("01*", pred_label, context)  
        pred_attribute = self.seek_pattern("23", pred_label, context) # B-attribute  I-attribute
        pred_value = self.seek_pattern("45", pred_label, context)  # B-value, I-value
        return pred_obj, pred_attribute, pred_value
        
        
        
    def predict(self, sentence:str):
        input_id = []
        for char in sentence:
            input_id.append(self.vocab.get(char, self.vocab['[UNK]']))

        with torch.no_grad():
            res = self.model.forward(torch.LongTensor([input_id]))[0] # shape = (batch_size, sen_len, class_num)
            res = torch.argmax(res, dim=-1) # shape = (batch_size, sen_len)
        object, attribute, value = self.decode(res, sentence)
        
        return object, attribute, value













if __name__ == "__main__":
    sl = SentenceLabel(Config, "model_output/epoch_15.pth")
    
    sentence = "马拉博是赤道几内亚首都，位于比奥科岛北端。"
    res = sl.predict(sentence)
    print(res)