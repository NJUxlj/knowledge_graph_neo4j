import neo4j   # pip install neo4j

from neo4j import GraphDatabase

import os  
from langchain.chains.retrieval_qa.base import RetrievalQA  
from langchain_community.vectorstores.neo4j_vector import Neo4jVector  
from langchain_openai import OpenAIEmbeddings, ChatOpenAI 





# 配置环境变量（建议通过.env文件加载）  
os.environ["NEO4J_URI"] = "bolt://localhost:7687"  
os.environ["NEO4J_USERNAME"] = "neo4j"  
os.environ["NEO4J_PASSWORD"] = "your_password"  
os.environ["OPENAI_API_KEY"] = "sk-xxx"


driver = GraphDatabase.driver(  
    os.environ["NEO4J_URI"],  
    auth=(os.environ["NEO4J_USERNAME"],   
          os.environ["NEO4J_PASSWORD"])  
)  



def query_neo4j():

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
        

if __name__ == "__main__":
    
    # 测试Neo4j连接
    test_connection()