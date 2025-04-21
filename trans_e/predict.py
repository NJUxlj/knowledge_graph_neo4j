# predict.py  
# 使用训练好的TransE模型进行预测  

import os  
import torch  
import numpy as np  
import argparse  
from model import TransE  
from loader import DataLoader  
from config import Config  

def load_model_and_data(model_path):  
    """  
    加载模型和数据  
    """  
    # 加载配置  
    config = Config()  
    
    # 加载数据映射  
    data_loader = DataLoader(config)  
    data_loader.load_mappings(model_path)  
    
    # 加载模型  
    model = TransE.load_model(model_path)  
    
    return model, data_loader  

def predict_tail_entities(model, data_loader, head, relation, top_k=10):  
    """  
    预测给定头实体和关系的前k个最可能的尾实体  
    """  
    # 确保输入是字符串  
    if isinstance(head, int):  
        head_id = head  
    else:  
        head_id = data_loader.entity2id.get(head)  
        if head_id is None:  
            print(f"Entity '{head}' not found in the knowledge graph")  
            return []  
    
    if isinstance(relation, int):  
        relation_id = relation  
    else:  
        relation_id = data_loader.relation2id.get(relation)  
        if relation_id is None:  
            print(f"Relation '{relation}' not found in the knowledge graph")  
            return []  
    
    # 预测所有尾实体的得分  
    tail_scores = model.predict_tail(head_id, relation_id)  
    
    # 获取得分最低的k个实体索引（得分越低表示越可能）  
    top_indices = np.argsort(tail_scores)[:top_k]  
    
    # 构建结果  
    results = []  
    for idx in top_indices:  
        entity = data_loader.id2entity[idx]  
        score = tail_scores[idx]  
        results.append((entity, score))  
    
    return results  

def predict_head_entities(model, data_loader, relation, tail, top_k=10):  
    """  
    预测给定关系和尾实体的前k个最可能的头实体  
    """  
    # 确保输入是字符串  
    if isinstance(tail, int):  
        tail_id = tail  
    else:  
        tail_id = data_loader.entity2id.get(tail)  
        if tail_id is None:  
            print(f"Entity '{tail}' not found in the knowledge graph")  
            return []  
    
    if isinstance(relation, int):  
        relation_id = relation  
    else:  
        relation_id = data_loader.relation2id.get(relation)  
        if relation_id is None:  
            print(f"Relation '{relation}' not found in the knowledge graph")  
            return []  
    
    # 预测所有头实体的得分  
    head_scores = model.predict_head(relation_id, tail_id)  
    
    # 获取得分最低的k个实体索引（得分越低表示越可能）  
    top_indices = np.argsort(head_scores)[:top_k]  
    
    # 构建结果  
    results = []  
    for idx in top_indices:  
        entity = data_loader.id2entity[idx]  
        score = head_scores[idx]  
        results.append((entity, score))  
    
    return results  

def predict_relations(model, data_loader, head, tail, top_k=10):  
    """  
    预测给定头实体和尾实体的前k个最可能的关系  
    """  
    # 确保输入是字符串  
    if isinstance(head, int):  
        head_id = head  
    else:  
        head_id = data_loader.entity2id.get(head)  
        if head_id is None:  
            print(f"Entity '{head}' not found in the knowledge graph")  
            return []  
    
    if isinstance(tail, int):  
        tail_id = tail  
    else:  
        tail_id = data_loader.entity2id.get(tail)  
        if tail_id is None:  
            print(f"Entity '{tail}' not found in the knowledge graph")  
            return []  
    
    # 获取所有关系的得分  
    scores = []  
    for relation_id in range(data_loader.n_relations):  
        # 获取关系嵌入  
        r_emb = model.relation_embeddings(torch.tensor([relation_id], dtype=torch.long))  
        
        # 获取头尾实体嵌入  
        h_emb = model.entity_embeddings(torch.tensor([head_id], dtype=torch.long))  
        t_emb = model.entity_embeddings(torch.tensor([tail_id], dtype=torch.long))  
        
        # 计算 h + r - t  
        score = h_emb + r_emb - t_emb  
        
        # 计算能量（距离）  
        if model.distance == 'L1':  
            energy = torch.norm(score, p=1, dim=1).item()  
        else:  # 'L2'  
            energy = torch.norm(score, p=2, dim=1).item()  
        
        scores.append((relation_id, energy))  
    
    # 按能量排序（能量越低表示越可能）  
    scores.sort(key=lambda x: x[1])  
    
    # 构建结果  
    results = []  
    for relation_id, score in scores[:top_k]:  
        relation = data_loader.id2relation[relation_id]  
        results.append((relation, score))  
    
    return results  

def main():  
    # 解析命令行参数  
    parser = argparse.ArgumentParser(description='Predict using trained TransE model')  
    parser.add_argument('--model_path', type=str, default='./models/', help='Path to the saved model')  
    parser.add_argument('--head', type=str, help='Head entity (for tail or relation prediction)')  
    parser.add_argument('--relation', type=str, help='Relation (for head or tail prediction)')  
    parser.add_argument('--tail', type=str, help='Tail entity (for head or relation prediction)')  
    parser.add_argument('--top_k', type=int, default=10, help='Number of top predictions to show')  
    parser.add_argument('--mode', type=str, choices=['tail', 'head', 'relation'], default='tail',  
                        help='Prediction mode: tail, head, or relation')  
    
    args = parser.parse_args()  
    
    # 加载模型和数据  
    model, data_loader = load_model_and_data(args.model_path)  
    
    # 根据模式进行预测  
    if args.mode == 'tail':  
        if args.head is None or args.relation is None:  
            print("Head entity and relation must be provided for tail prediction")  
            return  
        
        results = predict_tail_entities(model, data_loader, args.head, args.relation, args.top_k)  
        
        print(f"\nTop {len(results)} predicted tail entities for ({args.head}, {args.relation}, ?):")  
        for entity, score in results:  
            print(f"{entity}: {score:.4f}")  
    
    elif args.mode == 'head':  
        if args.relation is None or args.tail is None:  
            print("Relation and tail entity must be provided for head prediction")  
            return  
        
        results = predict_head_entities(model, data_loader, args.relation, args.tail, args.top_k)  
        
        print(f"\nTop {len(results)} predicted head entities for (?, {args.relation}, {args.tail}):")  
        for entity, score in results:  
            print(f"{entity}: {score:.4f}")  
    
    elif args.mode == 'relation':  
        if args.head is None or args.tail is None:  
            print("Head and tail entities must be provided for relation prediction")  
            return  
        
        results = predict_relations(model, data_loader, args.head, args.tail, args.top_k)  
        
        print(f"\nTop {len(results)} predicted relations for ({args.head}, ?, {args.tail}):")  
        for relation, score in results:  
            print(f"{relation}: {score:.4f}")  


if __name__ == "__main__":  
    main()  