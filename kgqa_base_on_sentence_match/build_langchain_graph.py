import neo4j   # pip install neo4j

from neo4j import GraphDatabase

from zhipuai import ZhipuAI

import os  
from langchain.chains.retrieval_qa.base import RetrievalQA  
from langchain_community.vectorstores.neo4j_vector import Neo4jVector  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # 使用开源嵌入模型  


from langchain_community.chat_models.zhipuai import ChatZhipuAI  
from langchain_core.embeddings import Embeddings 


from typing import Literal, Union, List




# 配置环境变量（建议通过.env文件加载）  
os.environ["NEO4J_URI"] = "bolt://localhost:7687"  
os.environ["NEO4J_USERNAME"] = "neo4j"  
os.environ["NEO4J_PASSWORD"] = "your_password"  
# os.environ["OPENAI_API_KEY"] = None


# os.environ["ZHIPUAI_API_KEY"] = "your_zhipu_api_key"  # 替换为智谱API密钥


driver = GraphDatabase.driver(  
    os.environ["NEO4J_URI"],  
    auth=(os.environ["NEO4J_USERNAME"],   
          os.environ["NEO4J_PASSWORD"])  
)  


class ZhipuAIEmbeddings(Embeddings):  
    """智谱Embedding服务封装类"""  
    def __init__(self, model="embedding-2"):  
        self.client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])  
        self.model = model  

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  
        # 官方接口支持批量处理（单次最多25条）  
        response = self.client.embeddings.create(  
            model=self.model,  
            input=texts  
        )  
        return [data.embedding for data in response.data]  

    def embed_query(self, text: str) -> List[float]:  
        return self.embed_documents([text])[0] 


def get_embedding(embed_type:Literal["zhipu","huggingface"]="zhipu", **kwargs):
    embedding = None
    if embed_type == "zhipu":
        embedding = ZhipuAIEmbeddings(model="embedding-2")
        
    else:
        # 使用开源嵌入模型（推荐智谱开源的embedding模型）  
        embedding = HuggingFaceEmbeddings(  
            model_name="BAAI/bge-large-zh-v1.5",  # 中文效果最佳  
            model_kwargs={'device': 'cpu'},        # 根据环境调整  
            encode_kwargs={'normalize_embeddings': True}  
        ) 
    
    return embedding



def query_neo4j_zhipu():
    # 创建Neo4j知识图谱连接（保持原有结构）  
    neo4j_vector = Neo4jVector.from_existing_graph(  
        embedding=get_embedding(),  
        url=os.environ["NEO4J_URI"],  
        username=os.environ["NEO4J_USERNAME"],  
        password=os.environ["NEO4J_PASSWORD"],  
        index_name="knowledge_graph",  
        node_label="Entity",  
        text_node_properties=["name", "description"],  
        embedding_node_property="embedding",  
    )  

    # 构建GLM-4问答链  
    qa_chain = RetrievalQA.from_chain_type(  
        llm=ChatZhipuAI(  
            model="glm-4-fast",   
            temperature=0.1,  
            api_key=os.environ["ZHIPUAI_API_KEY"]  
        ),  
        chain_type="stuff",  
        retriever=neo4j_vector.as_retriever(  
            search_type="similarity",  # 混合检索模式  
            search_kwargs={"k": 5}  
        ),  
        return_source_documents=True,  
        verbose=True  
    )  

    # 示例使用  
    question = "爱因斯坦的主要成就是什么？"  
    result = qa_chain.invoke({"query": question})  
    print(f"Answer: {result['result']}\nSources: {result['source_documents']}")  




def query_neo4j_openai():

    # 创建Neo4j知识图谱连接  
    neo4j_vector = Neo4jVector.from_existing_graph(  
        embedding=OpenAIEmbeddings(),  
        url=os.environ["NEO4J_URI"],  
        username=os.environ["NEO4J_USERNAME"],  
        password=os.environ["NEO4J_PASSWORD"],  
        index_name="knowledge_graph",  # 已存在的向量索引名称  
        node_label="Entity",          # 节点标签  
        text_node_properties=["name", "description"],  # 作为文本的节点属性  
        embedding_node_property="embedding",  # 存储embedding的属性名  
    )  

    # 构建问答链  
    qa_chain = RetrievalQA.from_chain_type(  
        llm=ChatOpenAI(temperature=0),  
        chain_type="stuff",  
        retriever=neo4j_vector.as_retriever(search_kwargs={"k": 3}),  
        return_source_documents=True,  
        verbose=True  
    )  

    # 示例使用  
    question = "爱因斯坦的主要成就是什么？"  
    result = qa_chain.invoke({"query": question})  
    print(f"Answer: {result['result']}\nSources: {result['source_documents']}")  



def test_connection():  
    with driver.session() as session:  
        result = session.run("MATCH (n) RETURN count(n) AS node_count")  
        print("当前知识图谱节点数量:", result.single()["node_count"])  
        
        
        
        
def test_zhipu_connection():  
    client = ZhipuAI(api_key=os.environ["ZHIPUAI_API_KEY"])  
    response = client.chat.completions.create(  
        model="glm-4-fast",  
        messages=[{"role": "user", "content": "你好"}],  
    )  
    print("API响应测试:", response.choices[0].message.content)  
        

if __name__ == "__main__":
    
    # 测试Neo4j连接
    test_connection()