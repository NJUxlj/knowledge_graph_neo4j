# config.py  
# TransE模型配置  

class Config:  
    # 数据集相关参数  
    data_path = "./data/"  # 数据集路径  
    
    # 模型超参数  
    embedding_dim = 50     # 实体和关系的嵌入维度  
    margin = 1.0           # 损失函数中的margin参数  
    learning_rate = 0.01   # 学习率  
    
    # 训练参数  
    batch_size = 128       # 批处理大小  
    epochs = 5          # 最大训练轮数  
    eval_freq = 10         # 每多少轮验证一次  论文中为1000
    early_stop = 5         # 提前停止的验证次数  
    
    # 损失函数的距离度量  
    distance = "L1"        # 可选 "L1" 或 "L2"  
    
    # 随机负采样参数  
    neg_ratio = 1          # 每个正样本对应的负样本数量  
    
    # 保存路径  
    model_save_path = "./output/"  # 模型保存路径  
    
    # 随机种子  
    seed = 42  