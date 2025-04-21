# evaluate.py  
# 评估TransE模型  

import numpy as np  
import time  
import torch  
from tqdm import tqdm  

def evaluate(model, data_loader, mode='valid', batch_size=100, filter_option='filtered'):  
    """  
    评估TransE模型，计算平均排名和hits@10  
    
    参数:  
        model: TransE模型  
        data_loader: 数据加载器  
        mode: 'valid' 或 'test'，指定评估的数据集  
        batch_size: 批处理大小  
        filter_option: 'raw' 或 'filtered'，指定评估方式  
        
    返回:  
        mean_rank: 平均排名  
        hits10: hits@10指标（前10名命中率）  
    """  
    model.eval()  # 设置为评估模式  
    
    if mode == 'valid':  
        triples = data_loader.valid_triples  
    else:  
        triples = data_loader.test_triples  
    
    all_ranks = []  
    head_ranks = []  
    tail_ranks = []  
    
    print(f"Evaluating on {mode} set ({len(triples)} triples)...")  
    start_time = time.time()  
    
    # 进度条  
    for i in tqdm(range(0, len(triples), batch_size)):  
        # 获取当前批次的三元组  
        batch_triples = triples[i:i+batch_size]  
        
        # 计算每个三元组的排名  
        for h, r, t in batch_triples:  
            # 替换尾实体进行评估  
            scores_t = model.predict_tail(h, r)  
            
            # 根据filter_option进行过滤  
            if filter_option == 'filtered':  
                # 过滤掉所有已知的正确三元组  
                for t_idx in data_loader.hr2t[(h, r)]:  
                    if t_idx != t:  # 保留当前正确的三元组  
                        scores_t[t_idx] = float('inf')  
            
            # 计算排名  
            rank_t = 1 + np.sum(scores_t < scores_t[t])  
            all_ranks.append(rank_t)  
            tail_ranks.append(rank_t)  
            
            # 替换头实体进行评估  
            scores_h = model.predict_head(r, t)  
            
            # 根据filter_option进行过滤  
            if filter_option == 'filtered':  
                # 过滤掉所有已知的正确三元组  
                for h_idx in data_loader.tr2h[(t, r)]:  
                    if h_idx != h:  # 保留当前正确的三元组  
                        scores_h[h_idx] = float('inf')  
            
            # 计算排名  
            rank_h = 1 + np.sum(scores_h < scores_h[h])  
            all_ranks.append(rank_h)  
            head_ranks.append(rank_h)  
    
    # 计算评估指标  
    all_ranks = np.array(all_ranks)  
    head_ranks = np.array(head_ranks)  
    tail_ranks = np.array(tail_ranks)  
    
    # 计算平均排名  
    mean_rank = np.mean(all_ranks)  
    mean_head_rank = np.mean(head_ranks)  
    mean_tail_rank = np.mean(tail_ranks)  
    
    # 计算hits@10  
    hits10 = np.mean(all_ranks <= 10) * 100  
    hits10_head = np.mean(head_ranks <= 10) * 100  
    hits10_tail = np.mean(tail_ranks <= 10) * 100  
    
    # 输出结果  
    print(f"Evaluation results ({filter_option}):")  
    print(f"Mean Rank: {mean_rank:.2f}, Hits@10: {hits10:.2f}%")  
    print(f"Mean Head Rank: {mean_head_rank:.2f}, Head Hits@10: {hits10_head:.2f}%")  
    print(f"Mean Tail Rank: {mean_tail_rank:.2f}, Tail Hits@10: {hits10_tail:.2f}%")  
    print(f"Evaluation time: {time.time() - start_time:.2f} seconds")  
    
    return mean_rank, hits10  

def detailed_evaluation(model, data_loader, mode='test'):  
    """  
    详细评估TransE模型，按关系类型分类  
    """  
    model.eval()  # 设置为评估模式  
    
    if mode == 'valid':  
        triples = data_loader.valid_triples  
    else:  
        triples = data_loader.test_triples  
    
    # 创建关系类型映射  
    relation_type = {}  
    
    # 计算每个关系的头尾基数  
    rel_head_avg = {}  
    rel_tail_avg = {}  
    
    for r in range(data_loader.n_relations):  
        head_per_tail = []  
        tail_per_head = []  
        
        for h, rel, t in data_loader.train_triples:  
            if rel == r:  
                head_per_tail.append((t, rel))  
                tail_per_head.append((h, rel))  
        
        if head_per_tail:  
            rel_head_avg[r] = len(set(head_per_tail)) / len(head_per_tail)  
        else:  
            rel_head_avg[r] = 0  
            
        if tail_per_head:  
            rel_tail_avg[r] = len(set(tail_per_head)) / len(tail_per_head)  
        else:  
            rel_tail_avg[r] = 0  
    
    # 确定关系类型  
    for r in range(data_loader.n_relations):  
        h_avg = rel_head_avg.get(r, 0)  
        t_avg = rel_tail_avg.get(r, 0)  
        
        if h_avg < 1.5 and t_avg < 1.5:  
            relation_type[r] = "1-TO-1"  
        elif h_avg < 1.5 and t_avg >= 1.5:  
            relation_type[r] = "1-TO-MANY"  
        elif h_avg >= 1.5 and t_avg < 1.5:  
            relation_type[r] = "MANY-TO-1"  
        else:  
            relation_type[r] = "MANY-TO-MANY"  
    
    # 按关系类型分类评估  
    results = {  
        "1-TO-1": {"head": [], "tail": []},  
        "1-TO-MANY": {"head": [], "tail": []},  
        "MANY-TO-1": {"head": [], "tail": []},  
        "MANY-TO-MANY": {"head": [], "tail": []}  
    }  
    
    print(f"Detailed evaluation on {mode} set ({len(triples)} triples)...")  
    
    for h, r, t in tqdm(triples):  
        r_type = relation_type.get(r, "UNKNOWN")  
        if r_type == "UNKNOWN":  
            continue  
        
        # 替换尾实体进行评估  
        scores_t = model.predict_tail(h, r)  
        
        # 过滤掉所有已知的正确三元组  
        for t_idx in data_loader.hr2t[(h, r)]:  
            if t_idx != t:  # 保留当前正确的三元组  
                scores_t[t_idx] = float('inf')  
        
        # 计算排名  
        rank_t = 1 + np.sum(scores_t < scores_t[t])  
        results[r_type]["tail"].append(rank_t <= 10)  
        
        # 替换头实体进行评估  
        scores_h = model.predict_head(r, t)  
        
        # 过滤掉所有已知的正确三元组  
        for h_idx in data_loader.tr2h[(t, r)]:  
            if h_idx != h:  # 保留当前正确的三元组  
                scores_h[h_idx] = float('inf')  
        
        # 计算排名  
        rank_h = 1 + np.sum(scores_h < scores_h[h])  
        results[r_type]["head"].append(rank_h <= 10)  
    
    # 计算并输出每种关系类型的hits@10  
    print("\nDetailed evaluation results (Hits@10):")  
    print(f"{'Relation Type':<15} {'Head (%)':<10} {'Tail (%)':<10}")  
    print("-" * 35)  
    
    for r_type in ["1-TO-1", "1-TO-MANY", "MANY-TO-1", "MANY-TO-MANY"]:  
        head_hits10 = np.mean(results[r_type]["head"]) * 100 if results[r_type]["head"] else 0  
        tail_hits10 = np.mean(results[r_type]["tail"]) * 100 if results[r_type]["tail"] else 0  
        print(f"{r_type:<15} {head_hits10:<10.2f} {tail_hits10:<10.2f}")  
    
    return results  