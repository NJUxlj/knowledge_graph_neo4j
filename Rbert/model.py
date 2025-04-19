# -*- coding: utf-8 -*-
import json
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
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"])
        self.hidden_size = self.bert.config.hidden_size
        self.cls_fc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.entity_fc_layer = nn.Linear(self.hidden_size, self.hidden_size)
        self.num_labels = self.config["num_labels"] if "num_labels" in self.config else self._get_num_labels()
        self.label_classifier = nn.Linear(self.hidden_size * 3, self.num_labels)
        self.activation = torch.tanh
        self.dropout = nn.Dropout(0.5)
        
        print("Rbert 初始化完成 ~~~")
        
    def _get_num_labels(self):
        num_labels = -1
        with open(file=self.config['schema_path'], mode = 'r', encoding="utf-8") as f:
            schema = json.load(f)
            
        if schema is None or schema == {}:
            raise ValueError("the schema file is empty, please check")
        
        num_labels = len(schema)
        
        return num_labels
    #entity mask 形如：    [0,0,1,1,0,0,..]
    def entity_average(self, hidden_output, e_mask):
        '''
        ### Function:
        对实体词求平均
        - 该函数用于于计算实体在BERT输出表示上的平均向量。这是关系抽取任务中的一个关键步骤，用于获取实体的语义表示
        
        ###Args:
        hidden_output: [batch_size, sentence_length, hidden_size]
        e_mask: [batch_size, sentence_length]  ， 这里在实际运行时，可以传e1_mask或e2_mask，这会帮助模型定位到一个特定的实体

        ###Returns:
        avg_vector: [batch_size, hidden_size]  含义：一个batch中，每个sequence对应的实体的embedding向量
        '''
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [batch_size, 1, sentence_length]

        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [batch_size, 1, sentence_length] * [b, sentence_length, hidden_size]  
        # = [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)  # [batch_size, hidden_size] -> batch中的每个sequence都对应一个实体的embedding

        avg_vector = sum_vector.float() / length_tensor.float()  # 除以实体词长度，做长度归一化
        return avg_vector
    
    def forward(self, input_ids, e1_mask, e2_mask, labels=None):
        outputs = self.bert(input_ids)
        sequence_output = outputs[0] #shape = [batch_size, sequence_length, hidden_size]
        pooled_output = outputs[1] # [CLS] shape = [batch_size, hidden_size]
        
        
        # 实体向量求平均
        e1_h = self.entity_average(sequence_output, e1_mask) # [batch_size, hidden_size]
        e2_h = self.entity_average(sequence_output, e2_mask) # [batch_size, hidden_size]
        
        # dropout
        e1_h = self.dropout(e1_h)
        e2_h = self.dropout(e2_h)
        pooled_output = self.dropout(pooled_output)
        
        #过线性层并激活
        pooled_output = self.activation(self.cls_fc_layer(pooled_output))
        e1_h = self.activation(self.entity_fc_layer(e1_h))
        e2_h = self.activation(self.entity_fc_layer(e2_h))
        
        # 拼接向量
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)  # shape = [batch_size, hidden_size * 3]
        logits = self.label_classifier(concat_h)  # shape = [batch_size, num_labels]
        
        # 计算损失
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),labels.view(-1))
            return loss
        else:
            return logits
    
    
    
    



def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    
    
    
    
    
    
if __name__ == "__main__":
    '''
    python -m Rbert.model
    '''
    from .config import Config
    model = RBert(Config)