# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List, Tuple

from config import Config

"""
数据加载
"""


class DataGenerator:
    def __init__(self, config, data_path):
        self.config = config
        self.path = data_path
        self.vocab = self.load_vocab(self.config["vocab_path"])
        self.config['vocab_size'] = len(self.vocab)
        self.sentences = []
        
        self.schema = {
            "B_object":0,
            "I_object":1,
            "B_attribute":2,
            "I_attribute":3,
            "B_value":4,
            "I_value":5,
            "O":6
        }
        
        
        self.config['class_num'] = len(self.schema)
        
        self.max_length = config["max_length"]
        self.neglect_count = 0
        
        self.load()
        
        print("超出设定最大长度的样本数量:%d, 占比:%.3f"%(self.exceed_max_length, self.exceed_max_length/len(self.data)))
        print("由于属性不存在于原文中，忽略%d条样本, 占比:%.3f"%(self.neglect_count, self.neglect_count / len(self.data)))

        
        
        
    def load(self):
        self.text_data = []
        self.data = []
        self.exceed_max_length = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                context = sample["context"]
                object = sample['object']
                attribute = sample['attribute']
                value = sample['value']
                
                # 对于基于序列标注的开放式抽取，如果目标属性（或关系）不出现在原文，则无法抽取
                if attribute not in context:
                    self.neglect_count += 1
                    continue
                
                self.text_data.append([context, object, attribute, value])
                input_id, label = self.process_sentence(context, object, attribute, value)
                self.data.append([torch.LongTensor(input_id), torch.LongTensor(label)])
        return
        
        
    def load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def process_sentence(self, context, object, attribute, value)->Tuple[List[int], List[int]]:
        '''
        我们默认， context中只包含一个三元组，且三元组的顺序为 object, attribute, value
        
        我们的目标就是将这个3元组对应的label识别出来。
        '''
        if len(context)> self.max_length:
            self.exceed_max_length += 1
            
        object_start = context.index(object)
        attribute_start = context.index(attribute)
        value_start = context.index(value)
        input_id = self.encode_sentence(context)
        
        assert len(context) == len(input_id) # 一个字符对应一个token
        
        # 初始化 label数组
        label = ["O"] * len(input_id)
        
        # 标记实体
        label[object_start] = "B_object"
        for i in range(object_start+1,  object_start+len(object)):
            label[i] = "I_object"

        # 标记属性
        label[attribute_start] = "B_attribute"
        for i in range(attribute_start+1, attribute_start+len(attribute)):
            label[i] = "I_attribute"

        # 标记属性值
        label[value_start] = "B_value"
        for i in range(value_start+1, value_start + len(value)):
            label[i] = "I_value"
        
        input_id = self.padding(input_id, 0)
        label = self.padding(label, -100)
        
        return input_id, label
        
        
    def encode_sentence(self, text:str, padding=False):
        '''
        将字符串转为 token id 序列
        '''
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, '[UNK]'))
            
        if padding:
            self.padding(input_id)
        return input_id
        
    
    
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id
    
    
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        return self.data[index]
    
    
    
    
    def load_schema(self, path)->Dict:
        with open(path, encoding = "utf-8") as f:
            schema = json.load(f)
        return schema