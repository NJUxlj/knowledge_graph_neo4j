# main.py  
# 主程序，用于训练和评估TransE模型  

import os  
import torch  
import numpy as np  
import random  
import argparse  
import time  
from torch.optim import Adam  

from config import Config  
from loader import DataLoader  
from model import TransE  
from evaluate import evaluate, detailed_evaluation  

def set_seed(seed):  
    """  
    设置随机种子以确保结果可复现  
    """  
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():  
        torch.cuda.manual_seed_all(seed)  

def main():  
    # 解析命令行参数  
    parser = argparse.ArgumentParser(description='Train and evaluate TransE model for knowledge graph embeddings')  
    parser.add_argument('--data_path', type=str, default='./data/', help='Path to the data directory')  
    parser.add_argument('--embedding_dim', type=int, default=50, help='Dimension of the embeddings')  
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for the loss function')  
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')  
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')  
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')  
    parser.add_argument('--distance', type=str, default='L1', choices=['L1', 'L2'], help='Distance metric (L1 or L2)')  
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  
    parser.add_argument('--eval_freq', type=int, default=10, help='Evaluation frequency (in epochs)')  
    parser.add_argument('--save_path', type=str, default='./models/', help='Path to save the model')  
    parser.add_argument('--early_stop', type=int, default=5, help='Early stopping patience')  
    
    args = parser.parse_args()  
    
    # 更新配置  
    config = Config()  
    config.data_path = args.data_path  
    config.embedding_dim = args.embedding_dim  
    config.margin = args.margin  
    config.learning_rate = args.learning_rate  
    config.batch_size = args.batch_size  
    config.epochs = args.epochs  
    config.distance = args.distance  
    config.seed = args.seed  
    config.eval_freq = args.eval_freq  
    config.model_save_path = args.save_path  
    config.early_stop = args.early_stop  
    
    # 设置随机种子  
    set_seed(config.seed)  
    
    # 创建保存目录  
    if not os.path.exists(config.model_save_path):  
        os.makedirs(config.model_save_path)  
    
    # 加载数据  
    data_loader = DataLoader(config)  
    n_entities, n_relations = data_loader.load_data()  
    
    # 保存映射  
    data_loader.save_mappings(config.model_save_path)  
    
    # 创建模型  
    model = TransE(  
        n_entities=n_entities,  
        n_relations=n_relations,  
        embedding_dim=config.embedding_dim,  
        margin=config.margin,  
        distance=config.distance  
    )  
    
    # 优化器  
    optimizer = Adam(model.parameters(), lr=config.learning_rate)  
    
    # 训练模型  
    train(model, data_loader, optimizer, config)  
    
    # 加载最佳模型  
    best_model = TransE.load_model(config.model_save_path)  
    
    # 在测试集上进行详细评估  
    print("\nEvaluating on test set (raw setting)...")  
    evaluate(best_model, data_loader, mode='test', filter_option='raw')  
    
    print("\nEvaluating on test set (filtered setting)...")  
    evaluate(best_model, data_loader, mode='test', filter_option='filtered')  
    
    # 在测试集上进行详细评估  
    detailed_evaluation(best_model, data_loader, mode='test')  

def train(model, data_loader, optimizer, config):  
    """  
    训练TransE模型  
    """  
    best_hits10 = 0  
    best_epoch = 0  
    no_improvement = 0  
    
    print("Start training...")  
    
    for epoch in range(config.epochs):  
        model.train()  # 设置为训练模式  
        total_loss = 0  
        start_time = time.time()  
        
        # 计算每个epoch需要的批次数  
        n_batches = len(data_loader.train_triples) // config.batch_size  
        
        for batch in range(n_batches):  
            # 获取批次数据  
            pos_triples, neg_triples = data_loader.get_batch(  
                batch_size=config.batch_size,  
                neg_ratio=config.neg_ratio  
            )  
            
            # 计算损失  
            loss = model.calculate_loss(pos_triples, neg_triples)  
            
            # 反向传播和优化  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            
            # 归一化实体嵌入  
            model.normalize_embeddings()  
            
            total_loss += loss.item()  
        
        # 计算平均损失  
        avg_loss = total_loss / n_batches  
        
        # 输出训练信息  
        print(f"Epoch {epoch+1}/{config.epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")  
        
        # 定期在验证集上评估  
        if (epoch + 1) % config.eval_freq == 0:  
            print(f"\nEvaluating at epoch {epoch+1}...")  
            mean_rank, hits10 = evaluate(model, data_loader, mode='valid', filter_option='filtered')  
            
            # 检查是否有改进  
            if hits10 > best_hits10:  
                best_hits10 = hits10  
                best_epoch = epoch + 1  
                no_improvement = 0  
                
                # 保存最佳模型  
                model.save_model(config.model_save_path)  
                print(f"New best model saved at epoch {epoch+1} with Hits@10: {hits10:.2f}%")  
            else:  
                no_improvement += 1  
                print(f"No improvement for {no_improvement} evaluations")  
            
            # 早停  
            if no_improvement >= config.early_stop:  
                print(f"Early stopping at epoch {epoch+1}")  
                break  
    
    print(f"Training finished. Best model at epoch {best_epoch} with Hits@10: {best_hits10:.2f}%")  


if __name__ == "__main__":  
    main()  