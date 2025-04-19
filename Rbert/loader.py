# -*- coding: utf-8 -*-
import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


from Rbert.config import Config


"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(self.config["pretrain_model_path"])
        self.sentences = []
        
        with open(self.config["schema_path"], 'r', encoding='utf-8') as f:
            self.attribute_schema = json.load(f)        

        self.config["num_labels"] = len(self.attribute_schema)
        
        self.max_length = self.config["max_length"]
        
        self.load()
        print("超出设定最大长度的样本数量:%d, 占比:%.3f"%(self.exceed_max_length, self.exceed_max_length/len(self.data)))
        print("由于文本截断，导致实体缺失的样本数量:%d, 占比%.3f"%(self.entity_disapper, self.entity_disapper/len(self.data)))
    
    
    
    
    def load(self):
        self.text_data = []
        self.data = []
        self.exceed_max_length = 0
        self.entity_disapper = 0  # 实体缺失的样本数量

        with open(self.path, encoding="utf8") as f:
            for line in f:
                sample = json.loads(line)
                context = sample["context"]
                object = sample["object"]  # 头实体
                attribute = sample["attribute"]  # 关系
                value = sample["value"]  # 尾实体
                
                if context == "" or value == "":
                    continue
                
                if attribute not in self.attribute_schema:
                    attribute = "UNRELATED"
                    
                try:
                    input_id, e1_mask, e2_mask, label = self.process_sentence(context, object, attribute, value)
                except IndexError:
                    self.entity_disapper += 1
                    continue
                
                self.data.append(
                    [torch.LongTensor(input_id),
                    torch.LongTensor(e1_mask),
                    torch.LongTensor(e2_mask),
                    torch.LongTensor([label])]
                )
    
    
    def process_sentence(self, context, object, attribute, value):
        '''
        ### Args:
            context: 文本
            object: 头实体
            attribute: 关系
            value: 尾实体
        '''
        if len(context) > self.max_length:
            self.exceed_max_length += 1
        
        # 移除了原来的+1偏移(不再需要，因为BERT的[CLS]已经包含在offset_mapping中) 
        # object_start = context.index(object) + 1 #因为bert的分词会在第一位增加[cls]，所以向后移动一位
        # value_start = context.index(value) + 1

        encoding = self.tokenizer(
            context, 
            max_length = self.max_length, 
            padding = "max_length",
            return_offsets_mapping=True,
            truncation=True
            )
        input_id = encoding["input_ids"]
        # 使用return_offsets_mapping=True获取每个token对应的原始字符位置
        # 通过offset_mapping准确找到实体对应的token位置
        offset_mapping = encoding["offset_mapping"]
        
        
        # 找到实体在原始文本中的位置
        object_start_char = context.index(object)
        object_end_char = object_start_char + len(object)
        value_start_char = context.index(value)
        value_end_char = value_start_char + len(value)
        
        
        attribute_label = self.attribute_schema[attribute]

        # 标记头实体
        e1_mask = [0] * len(input_id)
        
        '''
        在BERT的offset_mapping中，(start, end)表示每个token对应原始文本中的字符位置范围：

        start - 该token在原始文本中的起始字符索引（从0开始）
        end - 该token在原始文本中的结束字符索引（不包含）
        
        例如：

        对于文本"我爱北京"，分词后可能得到：
        "我" → (0,1)
        "爱" → (1,2)
        "北" → (2,3)
        "京" → (3,4)
        
        特殊token的处理：

        [CLS]和[SEP]等特殊token的offset_mapping通常是(0,0)
        被截断或填充的部分也是(0,0)
        在实体标记逻辑中，我们通过比较实体的字符范围(object_start_char, object_end_char)和每个token的(start,end)来确定哪些token属于该实体。
                
        '''
        for i, (start, end) in enumerate(offset_mapping):
            # if start <= object_start_char < end or start < object_end_char <= end or (object_start_char <= start and end <= object_end_char):
            #     e1_mask[i] = 1
            
            if (object_start_char <= start and end-1 <= object_end_char-1):
                e1_mask[i] = 1
                
        assert sum(e1_mask) >= 1, (object_start_char, object, e1_mask, list(range(object_start_char, object_start_char+len(object))), context)   # 作用：确保头实体被正确标记
        # 标记尾实体
        e2_mask = [0] * len(input_id)
        
        for i, (start, end) in enumerate(offset_mapping):
            # if start <= value_start_char < end or start < value_end_char <= end or (value_start_char <= start and end <= value_end_char):
            #     e2_mask[i] = 1
            
            if (value_start_char <= start and end-1 <= value_end_char-1):
                e2_mask[i] = 1
            
        return input_id, e1_mask, e2_mask, attribute_label  


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, "r", encoding = "utf-8") as f:
            schema = json.load(f)
        return schema




def load_data(data_path, config, shuffle):
    dg = DataGenerator(data_path, config)
    
    dl = DataLoader(dataset = dg, batch_size = config["batch_size"], shuffle = shuffle)

    return dl


if __name__ == "__main__":
    '''
    python -m Rbert.loader
    '''
    
    dg = DataGenerator(Config["train_data_path"], Config)
    
    print(dg[1])



