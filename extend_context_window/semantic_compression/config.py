"""  
配置文件，包含模型和数据处理的参数  
"""  

class Config:  
    # 文本处理配置  
    MAX_INPUT_LENGTH = 32000  # 最大输入长度  
    MAX_OUTPUT_LENGTH = 4096  # 期望的压缩后长度  
    
    # 分块配置  
    SENTENCE_CHUNK_SIZE = 3  # 每个初始块的句子数量  
    MIN_BLOCK_SIZE = 2  # 主题块的最小句子数  
    MAX_BLOCK_SIZE = 20  # 主题块的最大句子数  
    
    # 主题发现配置  
    CLUSTERING_METHOD = "community_detection"  # 可选: "community_detection", "agglomerative", "kmeans"  
    MIN_SIMILARITY = 0.5  # 构建图时的最小相似度阈值  
    MAX_CLUSTERS = 30  # 最大聚类数量  
    
    # 模型配置  
    SENTENCE_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"  # 句子编码器  
    SUMMARIZER = "facebook/bart-large-cnn"  # 摘要模型  
    LLM = "meta-llama/Llama-2-7b-chat-hf"  # 大型语言模型  
    
    # 硬件配置  
    DEVICE = "cuda"  # 使用的设备 "cuda" 或 "cpu"  
    
    # 评估配置  
    EVAL_METRICS = ["rouge", "bertscore", "perplexity"]  # 评估指标  