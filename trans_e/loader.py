# loader.py  
# 数据加载和预处理  

import os  
import numpy as np  
import pandas as pd  
from collections import defaultdict  
import pickle  
import random  

class DataLoader:  
    def __init__(self, config):  
        self.config = config  
        self.data_path = config.data_path  
        
        # 实体、关系和三元组的集合  
        self.entities = set()  
        self.relations = set()  
        self.train_triples = []  
        self.valid_triples = []  
        self.test_triples = []  
        
        # 实体和关系的映射，用于将字符串转换为索引  
        self.entity2id = {}  
        self.relation2id = {}  
        self.id2entity = {}  
        self.id2relation = {}  
        
        # 所有三元组集合（用于filtered评估）  
        self.all_true_triples = set()  
        
        # 实体数量和关系数量  
        self.n_entities = 0  
        self.n_relations = 0  
        
        # 头实体-关系对应的尾实体字典和尾实体-关系对应的头实体字典  
        self.hr2t = defaultdict(set)  
        self.tr2h = defaultdict(set)  
        
    def load_data(self, files=None):  
        """  
        加载数据集，包括训练集、验证集和测试集  
        """  
        if files is None:  
            # 默认文件名  
            files = {  
                'train': 'train.txt',  
                'valid': 'valid.txt',  
                'test': 'test.txt'  
            }  
        
        print("Loading data...")  
        
        # 加载训练集  
        self.train_triples = self._read_file(os.path.join(self.data_path, files['train']))  
        
        # 加载验证集  
        self.valid_triples = self._read_file(os.path.join(self.data_path, files['valid']))  
        
        # 加载测试集  
        self.test_triples = self._read_file(os.path.join(self.data_path, files['test']))  
        
        # 创建实体和关系的映射  
        self._create_mappings()  
        
        # 将字符串类型的三元组转换为ID类型  
        self.train_triples = self._str2id(self.train_triples)  
        self.valid_triples = self._str2id(self.valid_triples)  
        self.test_triples = self._str2id(self.test_triples)  
        
        # 获取所有三元组（用于filtered评估）  
        self.all_true_triples = set(self.train_triples + self.valid_triples + self.test_triples)  
        
        # 创建hr2t和tr2h字典  
        self._create_lookup()  
        
        print(f"Data loaded: {self.n_entities} entities, {self.n_relations} relations")  
        print(f"Train: {len(self.train_triples)}, Valid: {len(self.valid_triples)}, Test: {len(self.test_triples)}")  
        
        return self.n_entities, self.n_relations  
    
    def _read_file(self, file_path):  
        """  
        读取文件，每行格式：头实体 关系 尾实体  
        """  
        triples = []  
        with open(file_path, 'r', encoding='utf-8') as f:  
            for line in f:  
                h, r, t = line.strip().split()  
                triples.append((h, r, t))  
                self.entities.add(h)  
                self.entities.add(t)  
                self.relations.add(r)  
        return triples  
    
    def _create_mappings(self):  
        """  
        创建实体和关系的ID映射  
        """  
        # 为实体创建映射  
        self.entity2id = {entity: idx for idx, entity in enumerate(sorted(self.entities))}  
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}  
        
        # 为关系创建映射  
        self.relation2id = {relation: idx for idx, relation in enumerate(sorted(self.relations))}  
        self.id2relation = {idx: relation for relation, idx in self.relation2id.items()}  
        
        # 更新实体和关系的数量  
        self.n_entities = len(self.entity2id)  
        self.n_relations = len(self.relation2id)  
    
    def _str2id(self, triples):  
        """  
        将字符串类型的三元组转换为ID类型  
        """  
        return [(self.entity2id[h], self.relation2id[r], self.entity2id[t]) for h, r, t in triples]  
    
    def _create_lookup(self):  
        """  
        创建hr2t和tr2h字典，用于快速检索  
        """  
        for h, r, t in self.train_triples + self.valid_triples + self.test_triples:  
            self.hr2t[(h, r)].add(t)  
            self.tr2h[(t, r)].add(h)  
    
    def get_batch(self, batch_size=None, neg_ratio=None):  
        """  
        获取训练批次，包括正样本和负样本  
        """  
        if batch_size is None:  
            batch_size = self.config.batch_size  
        if neg_ratio is None:  
            neg_ratio = self.config.neg_ratio  
        
        # 随机选择正样本  
        pos_triples = random.sample(self.train_triples, batch_size)  
        
        # 为每个正样本生成neg_ratio个负样本  
        neg_triples = []  
        
        for h, r, t in pos_triples:  
            # 对于每个正样本，随机决定是替换头实体还是尾实体  
            for _ in range(neg_ratio):  
                is_head = random.random() < 0.5  
                
                if is_head:  # 替换头实体  
                    # 随机选择一个头实体  
                    while True:  
                        h_neg = random.randint(0, self.n_entities - 1)  
                        if (h_neg, r, t) not in self.all_true_triples:  
                            break  
                    neg_triples.append((h_neg, r, t))  
                else:  # 替换尾实体  
                    # 随机选择一个尾实体  
                    while True:  
                        t_neg = random.randint(0, self.n_entities - 1)  
                        if (h, r, t_neg) not in self.all_true_triples:  
                            break  
                    neg_triples.append((h, r, t_neg))  
        
        return pos_triples, neg_triples  
    
    def save_mappings(self, save_path):  
        """  
        保存实体和关系的映射  
        """  
        if not os.path.exists(save_path):  
            os.makedirs(save_path)  
            
        with open(os.path.join(save_path, 'entity2id.pkl'), 'wb') as f:  
            pickle.dump(self.entity2id, f)  
        
        with open(os.path.join(save_path, 'relation2id.pkl'), 'wb') as f:  
            pickle.dump(self.relation2id, f)  
    
    def load_mappings(self, load_path):  
        """  
        加载实体和关系的映射  
        """  
        with open(os.path.join(load_path, 'entity2id.pkl'), 'rb') as f:  
            self.entity2id = pickle.load(f)  
            self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}  
            self.n_entities = len(self.entity2id)  
        
        with open(os.path.join(load_path, 'relation2id.pkl'), 'rb') as f:  
            self.relation2id = pickle.load(f)  
            self.id2relation = {idx: relation for relation, idx in self.relation2id.items()}  
            self.n_relations = len(self.relation2id)  