# model.py  
# TransE模型实现  

import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import torch.optim as optim  
import os  

class TransE(nn.Module):  
    def __init__(self, n_entities, n_relations, embedding_dim, margin=1.0, distance='L1'):  
        """  
        TransE模型初始化  
        
        参数:  
            n_entities: 实体数量  
            n_relations: 关系数量  
            embedding_dim: 嵌入维度  
            margin: 间隔参数  
            distance: 距离度量方式 ('L1' 或 'L2')  
        """  
        super(TransE, self).__init__()  
        
        self.n_entities = n_entities  
        self.n_relations = n_relations  
        self.embedding_dim = embedding_dim  
        self.margin = margin  
        self.distance = distance  
        
        # 初始化实体和关系的嵌入  
        # 按照论文描述，初始化方式采用Glorot初始化  
        bound = 6 / np.sqrt(embedding_dim)  
        
        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)  
        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)  
        
        # 初始化权重  
        nn.init.uniform_(self.entity_embeddings.weight, -bound, bound)  
        nn.init.uniform_(self.relation_embeddings.weight, -bound, bound)  
        
        # 归一化实体嵌入  
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)  
    
    def forward(self, h, r, t):  
        """  
        计算模型的能量（相似度得分）  
        较低的能量表示三元组更可能为真  
        
        参数:  
            h: 头实体索引  
            r: 关系索引  
            t: 尾实体索引  
            
        返回:  
            能量值  
        """  
        # 获取嵌入向量  
        h_emb = self.entity_embeddings(h)  
        r_emb = self.relation_embeddings(r)  
        t_emb = self.entity_embeddings(t)  
        
        # 计算 h + r - t  
        score = h_emb + r_emb - t_emb  
        
        # 计算距离  
        if self.distance == 'L1':  
            energy = torch.norm(score, p=1, dim=1)  
        else:  # 'L2'  
            energy = torch.norm(score, p=2, dim=1)  
            
        return energy  
    
    def normalize_embeddings(self):  
        """  
        归一化实体嵌入的范数  
        """  
        self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)  
    
    def calculate_loss(self, pos_triples, neg_triples):  
        """  
        计算损失函数  
        
        参数:  
            pos_triples: 正样本三元组 [(h, r, t), ...]  
            neg_triples: 负样本三元组 [(h', r, t'), ...]  
            
        返回:  
            损失值  
        """  
        # 转换为张量  
        pos_h = torch.tensor([triple[0] for triple in pos_triples], dtype=torch.long)  
        pos_r = torch.tensor([triple[1] for triple in pos_triples], dtype=torch.long)  
        pos_t = torch.tensor([triple[2] for triple in pos_triples], dtype=torch.long)  
        
        neg_h = torch.tensor([triple[0] for triple in neg_triples], dtype=torch.long)  
        neg_r = torch.tensor([triple[1] for triple in neg_triples], dtype=torch.long)  
        neg_t = torch.tensor([triple[2] for triple in neg_triples], dtype=torch.long)  
        
        # 计算正样本和负样本的能量  
        pos_energy = self.forward(pos_h, pos_r, pos_t)  
        neg_energy = self.forward(neg_h, neg_r, neg_t)  
        
        # 计算损失: [margin + d(h+r, t) - d(h'+r, t')]_+  
        # 其中 [x]_+ 表示 max(0, x)  
        loss = torch.mean(F.relu(self.margin + pos_energy - neg_energy))  
        
        return loss  
    
    def predict(self, h, r, t=None):  
        """  
        预测给定的头实体和关系的尾实体得分  
        
        参数:  
            h: 头实体索引  
            r: 关系索引  
            t: 如果提供，则只计算给定尾实体的得分；否则计算所有实体作为尾实体的得分  
            
        返回:  
            所有实体作为尾实体的得分，或单个尾实体的得分  
        """  
        h_emb = self.entity_embeddings(torch.tensor([h], dtype=torch.long))  
        r_emb = self.relation_embeddings(torch.tensor([r], dtype=torch.long))  
        
        # 如果提供了尾实体，只计算该尾实体的得分  
        if t is not None:  
            t_emb = self.entity_embeddings(torch.tensor([t], dtype=torch.long))  
            score = h_emb + r_emb - t_emb  
            
            if self.distance == 'L1':  
                return torch.norm(score, p=1, dim=1).item()  
            else:  # 'L2'  
                return torch.norm(score, p=2, dim=1).item()  
        
        # 否则计算所有实体作为尾实体的得分  
        all_entities = torch.arange(self.n_entities, dtype=torch.long)  
        all_t_emb = self.entity_embeddings(all_entities)  
        
        score = h_emb + r_emb - all_t_emb  
        
        if self.distance == 'L1':  
            return torch.norm(score, p=1, dim=1).numpy()  
        else:  # 'L2'  
            return torch.norm(score, p=2, dim=1).numpy()  
    
    def predict_tail(self, h, r):  
        """  
        预测给定头实体和关系的尾实体排名  
        
        参数:  
            h: 头实体索引  
            r: 关系索引  
            
        返回:  
            所有实体作为尾实体的得分  
        """  
        return self.predict(h, r)  
    
    def predict_head(self, r, t):  
        """  
        预测给定关系和尾实体的头实体得分  
        
        参数:  
            r: 关系索引  
            t: 尾实体索引  
            
        返回:  
            所有实体作为头实体的得分  
        """  
        t_emb = self.entity_embeddings(torch.tensor([t], dtype=torch.long))  
        r_emb = self.relation_embeddings(torch.tensor([r], dtype=torch.long))  
        
        # 计算所有实体作为头实体的得分  
        all_entities = torch.arange(self.n_entities, dtype=torch.long)  
        all_h_emb = self.entity_embeddings(all_entities)  
        
        # h + r = t  =>  h = t - r  
        score = t_emb - r_emb - all_h_emb  
        
        if self.distance == 'L1':  
            return torch.norm(score, p=1, dim=1).numpy()  
        else:  # 'L2'  
            return torch.norm(score, p=2, dim=1).numpy()  
    
    def save_model(self, save_path):  
        """  
        保存模型  
        """  
        if not os.path.exists(save_path):  
            os.makedirs(save_path)  
            
        torch.save({  
            'entity_embeddings': self.entity_embeddings.state_dict(),  
            'relation_embeddings': self.relation_embeddings.state_dict(),  
            'model_config': {  
                'n_entities': self.n_entities,  
                'n_relations': self.n_relations,  
                'embedding_dim': self.embedding_dim,  
                'margin': self.margin,  
                'distance': self.distance  
            }  
        }, os.path.join(save_path, 'transe_model.pt'))  
    
    @classmethod  
    def load_model(cls, load_path):  
        """  
        加载模型  
        """  
        checkpoint = torch.load(os.path.join(load_path, 'transe_model.pt'))  
        config = checkpoint['model_config']  
        
        model = cls(  
            n_entities=config['n_entities'],  
            n_relations=config['n_relations'],  
            embedding_dim=config['embedding_dim'],  
            margin=config['margin'],  
            distance=config['distance']  
        )  
        
        model.entity_embeddings.load_state_dict(checkpoint['entity_embeddings'])  
        model.relation_embeddings.load_state_dict(checkpoint['relation_embeddings'])  
        
        return model  