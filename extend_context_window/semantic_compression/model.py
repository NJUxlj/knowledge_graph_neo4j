"""  
实现核心的语义压缩算法  
"""  

import torch  
import numpy as np  
import networkx as nx  
from typing import List, Dict, Any, Tuple, Optional  
from sentence_transformers import SentenceTransformer  
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    BartForConditionalGeneration,
) 
from community import best_partition    # pip install python-louvain
from sklearn.cluster import AgglomerativeClustering, KMeans  
import logging  

from .config import Config

class SemanticCompressor:  
    """语义压缩模型类"""  
    
    def __init__(self, config:Config):  
        self.config = config  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
        
        # 初始化模型  
        self.logger.info(f"加载句子编码器: {config.SENTENCE_ENCODER}")  
        self.sentence_encoder = SentenceTransformer(config.SENTENCE_ENCODER)  
        self.sentence_encoder.to(config.DEVICE)  
        
        self.logger.info(f"加载摘要模型: {config.SUMMARIZER}")  
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(config.SUMMARIZER)  
        self.summarizer_model:BartForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER)  
        self.summarizer_model.to(config.DEVICE)  
    
    def encode_chunks(self, chunks: List[str]) -> torch.Tensor:  
        """  
        对文本块进行编码  
        
        Args:  
            chunks: 文本块列表  
            
        Returns:  
            embeddings: 文本块的嵌入向量  shape = [len(chunks), 768]
        """  
        self.logger.info(f"编码 {len(chunks)} 个文本块")  
        embeddings = self.sentence_encoder.encode(  
            chunks,   
            convert_to_tensor=True,   
            show_progress_bar=True,  
            device=self.config.DEVICE  
        )  # shape = [len(chunks), 768]
        return embeddings  
    
    def build_similarity_graph(self, embeddings: torch.Tensor) -> nx.Graph:  
        """  
        构建相似度图  
        
        Args:  
            embeddings: 文本块的嵌入向量  shape= [len(chunks), 768]
            
        Returns:  
            相似度图  
        """  
        self.logger.info("构建相似度图")  
        # 计算余弦相似度矩阵  
        # [len(chunks), 768] * [768, len(chunks)] = [len(chunks), len(chunks)]
        # 得到了一个相似度矩阵，其中每个元素表示两个文本块之间的相似度
        cos_sim = torch.mm(embeddings, embeddings.transpose(0, 1))  
        cos_sim = cos_sim.cpu().numpy()  
        
        # 构建图  
        G = nx.Graph()  
        
        # 添加节点  
        for i in range(len(cos_sim)):  # 有几个文本块就添加几个节点
            G.add_node(i)  
        
        # 添加边（只连接相似度高于阈值的节点）  
        for i in range(len(cos_sim)):  
            for j in range(i+1, len(cos_sim)):  
                similarity = cos_sim[i, j]  
                if similarity > self.config.MIN_SIMILARITY:  
                    G.add_edge(i, j, weight=float(similarity))  
        
        self.logger.info(f"相似度图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边")  
        return G  
    
    def discover_topics(self, G: nx.Graph, chunks: List[str]) -> List[List[int]]:  
        """  
        发现主题结构  
        
        Args:  
            G: 相似度图  
            chunks: 文本块列表  
            
        Returns:  
            主题块的索引列表  
        """  
        method = self.config.CLUSTERING_METHOD  
        self.logger.info(f"使用 {method} 方法进行主题发现")  
        
        if method == "community_detection":  
            # 使用Louvain社区检测  
            partition = best_partition(G)  
            topics = {}  
            for node, community_id in partition.items(): # 遍历每个《节点，聚类id》对 
                if community_id not in topics:  
                    topics[community_id] = []  
                topics[community_id].append(node)  
            
            # 转换为列表  
            topic_clusters = list(topics.values())  # type = List[List[node_id]]
            
        elif method == "agglomerative":  
            # 获取邻接矩阵，矩阵元素值代表节点间的相似度权重
            adjacency_matrix = nx.to_numpy_array(G)  # shape = [len(chunks), len(chunks)]
            
            # 使用层次聚类 
            '''
            这是一种**自底向上**的聚类方法，初始时每个节点是一个簇，然后逐步合并最相似的簇
                使用平均链接(average linkage)能平衡单链接(过于敏感)和全链接(过于保守)的特点
                适合处理图结构数据，能保留原始相似度关系
                时间复杂度较高(O(n³))，适合中小规模数据
            ''' 
            clustering = AgglomerativeClustering(  
                n_clusters=min(self.config.MAX_CLUSTERS, len(chunks)),  
                affinity='precomputed',  # 表示直接使用计算好的距离矩阵
                linkage='average'  # 使用平均链接算法计算类间距离
            )  
            # 将相似度矩阵转换为距离矩阵（相似度越高，距离越小）
            # 这是层次聚类需要的输入形式，因为聚类算法是基于距离而非相似度
            distance_matrix = 1 - adjacency_matrix  
            labels = clustering.fit_predict(distance_matrix)  # labels,type = ndarray, shape = [len(chunks)], 每个node index 对应一个label
            
            # 整理聚类结果  
            topics = {}  
            # 将聚类标签转换为{聚类ID: [节点列表]}的字典结构, 最终返回按聚类分组的节点索引列表
            for node, label in enumerate(labels):  
                if label not in topics:  
                    topics[label] = []  
                topics[label].append(node)  
            
            topic_clusters = list(topics.values())  # type = List[List[node_id]]
            
        elif method == "kmeans":  
            # 使用KMeans聚类  
            embeddings = np.array([np.array(G.nodes[i]['embedding']) for i in range(len(G.nodes))])  
            clustering = KMeans(  
                n_clusters=min(self.config.MAX_CLUSTERS, len(chunks)),  
                random_state=42  
            )  
            labels = clustering.fit_predict(embeddings)  # labels,type = ndarray, shape = [len(chunks)], 每个node index 对应一个label
            
            # 整理聚类结果  
            topics = {}  
            for node, label in enumerate(labels):  
                if label not in topics:  
                    topics[label] = []  
                topics[label].append(node)  
            
            topic_clusters = list(topics.values())  
        
        else:  
            raise ValueError(f"不支持的聚类方法: {method}")  
        
        self.logger.info(f"主题发现完成, 共 {len(topic_clusters)} 个主题")  
        return topic_clusters  
    
    def sort_topic_clusters(self, topic_clusters: List[List[int]]) -> List[List[int]]:  
        """  
        按原始文档顺序对主题块进行排序  
        
        Args:  
            topic_clusters: 主题块的索引列表  
            
        Returns:  
            排序后的主题块索引列表  
        """  
        # 按照每个主题块中第一个块的索引排序  
        return sorted(topic_clusters, key=lambda x: min(x))  
    
    def compress_topic(self, topic_indices: List[int], chunks: List[str]) -> str:  
        """  
        压缩单个主题  
        
        Args:  
            topic_indices: 属于该主题的块索引  
            chunks: 所有文本块  
            
        Returns:  
            压缩后的主题文本  
        """  
        # 获取该主题的所有文本  
        topic_text = " ".join([chunks[i] for i in topic_indices])  
        
        # 使用BART模型进行压缩  
        inputs = self.summarizer_tokenizer(  
            topic_text,   
            max_length=1024,   
            truncation=True,   
            return_tensors="pt"  
        ).to(self.config.DEVICE)  
        
        # 生成摘要  
        with torch.no_grad():  
            output_ids = self.summarizer_model.generate(  
                inputs["input_ids"],  
                max_length=250,  
                min_length=50,  
                length_penalty=2.0,  
                num_beams=4,  
                early_stopping=True  
            )  
        
        compressed_text = self.summarizer_tokenizer.decode(output_ids[0], skip_special_tokens=True)  
        
        return compressed_text  
    
    def compress_content(self, chunks: List[str]) -> Dict[str, Any]:  
        """  
        执行完整的语义压缩过程  
        
        Args:  
            chunks: 初始文本块列表  
            
        Returns:  
            包含压缩结果的字典  
        """  
        # 步骤1: 嵌入文本块  
        embeddings = self.encode_chunks(chunks)  
        
        # 步骤2: 构建相似度图  
        graph = self.build_similarity_graph(embeddings)  
        
        # 步骤3: 发现主题结构  
        topic_clusters:List[List[int]] = self.discover_topics(graph, chunks)  
        
        # 步骤4: 排序主题块  
        sorted_topic_clusters:List[List[int]] = self.sort_topic_clusters(topic_clusters)  
        
        # 步骤5: 压缩每个主题  
        self.logger.info("开始压缩主题")  
        compressed_topics:List[str] = []  
        topic_map:List[Dict[str, Any]] = []  
        
        for topic_idx, topic_indices in enumerate(sorted_topic_clusters):  
            self.logger.info(f"压缩主题 {topic_idx+1}/{len(sorted_topic_clusters)}, 包含 {len(topic_indices)} 个块")  
            compressed_text = self.compress_topic(topic_indices, chunks)  
            compressed_topics.append(compressed_text)  
            
            # 记录映射关系  
            for chunk_idx in topic_indices:
                topic_map.append({  
                    "chunk_idx": chunk_idx,  
                    "topic_idx": topic_idx,  
                    "original_text": chunks[chunk_idx],  
                    "compressed_text": compressed_text  
                })  
        
        # 步骤6: 重组压缩内容  
        compressed_content = " ".join(compressed_topics)  
        
        self.logger.info(f"压缩完成, 原始块数: {len(chunks)}, 压缩后主题数: {len(compressed_topics)}")  
        self.logger.info(f"压缩比: {len(compressed_content)/sum([len(c) for c in chunks]):.2f}")  
        
        return {  
            "compressed_content": compressed_content,  
            "compressed_topics": compressed_topics,  
            "topic_clusters": sorted_topic_clusters,  
            "topic_map": topic_map,  
            "original_chunks": chunks,  
            "compression_ratio": len(compressed_content)/sum([len(c) for c in chunks])  
        }  
        
        
        

def draw_nx_graph(G:nx.Graph, best_partition):
    """
    绘制nx图
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap
    pos = nx.spring_layout(G)  # 使用spring布局
    
    # color the nodes according to their partitions
    cmap = cm.get_cmap('viridis', max(best_partition.values())+1)  # 使用tab20颜色映射

    nx.draw_networkx_nodes(G, pos, best_partition.keys(), node_size=40,
                            cmap=cmap, node_color=list(best_partition.values()))
    
    
    nx.draw_networkx_edges(G, pos, alpha=0.5)  # 绘制边
    
    plt.show()

if __name__ == "__main__":
    pass