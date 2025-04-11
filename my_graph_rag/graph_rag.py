
import os
import tiktoken
from sentence_transformers import SentenceTransformer
from py2neo import Graph, Node, Relationship
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from transformers import AutoModelForSequenceClassification, BertTokenizer
import zhipuai

# 配置信息
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "password")
ZHIPU_API_KEY = "your_zhipu_key"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

class EnhancedGraphRAG:
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.ner_model = pipeline("ner", model="dslim/bert-base-NER")
        self.rel_tokenizer = BertTokenizer.from_pretrained('joeddav/xlm-roberta-large-softmax-batch')
        self.rel_model = AutoModelForSequenceClassification.from_pretrained('joeddav/xlm-roberta-large-softmax-batch')
        self.graph = Graph(NEO4J_URI, auth=NEO4J_AUTH)
        
        # 初始化图模式
        self._init_graph_schema()

    def _init_graph_schema(self):
        """创建图数据库约束"""
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Relationship) REQUIRE r.id IS UNIQUE")

    def split_text(self, text, chunk_size=50):
        """使用tiktoken进行精确分块"""
        tokens = self.tokenizer.encode(text)
        chunks = [self.tokenizer.decode(tokens[i:i+chunk_size]) 
                for i in range(0, len(tokens), chunk_size)]
        return chunks

    def extract_entities_zhipu(self, text):
        """使用智谱API进行实体识别"""
        zhipuai.api_key = ZHIPU_API_KEY
        response = zhipuai.model_api.invoke(
            model="chatglm_pro",
            prompt=[{"role": "user", "content": f"实体识别：{text}"}]
        )
        return self._parse_zhipu_response(response)

    def extract_entities_hf(self, text):
        """使用HuggingFace模型进行实体识别"""
        ner_results = self.ner_model(text)
        entities = []
        current_entity = {"text": "", "type": ""}
        
        for token in ner_results:
            if token['entity'].startswith('B-'):
                if current_entity["text"]:
                    entities.append(current_entity)
                current_entity = {
                    "text": token['word'],
                    "type": token['entity'][2:]
                }
            elif token['entity'].startswith('I-'):
                current_entity["text"] += token['word']
        
        return [f"{e['text']}({e['type']})" for e in entities]

    def extract_relationships(self, text, entities):
        """使用RBERT模型进行关系抽取"""
        inputs = self.rel_tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.rel_model(**inputs)
        predictions = outputs.logits.argmax(-1).item()
        
        # 简化的关系映射表
        RELATION_MAP = {
            0: "作者",
            1: "讨论",
            2: "属于"
        }
        return RELATION_MAP.get(predictions, "未知关系")

    def create_neo4j_nodes(self, entities):
        """创建Neo4j节点并生成嵌入"""
        tx = self.graph.begin()
        
        for entity in entities:
            node = Node("Entity",
                      id=entity['id'],
                      title=entity['title'],
                      type=entity['type'],
                      description=entity['description'],
                      embedding=EMBEDDING_MODEL.encode(entity['description'])
                      )
            tx.create(node)
            
        tx.commit()

    def create_neo4j_relationships(self, relationships):
        """创建Neo4j关系"""
        tx = self.graph.begin()
        
        for rel in relationships:
            source = self.graph.nodes.match(id=rel['source']).first()
            target = self.graph.nodes.match(id=rel['target']).first()
            
            relationship = Relationship(source, rel['type'], target,
                                      weight=rel['weight'],
                                      text_units=rel['text_unit_ids'])
            tx.create(relationship)
            
        tx.commit()

    def generate_embeddings(self, text):
        """生成实体嵌入"""
        return EMBEDDING_MODEL.encode(text)

# 示例用法 ################################
if __name__ == "__main__":
    rag = EnhancedGraphRAG()
    
    # 1. 处理文本
    text = "《大数据时代》由维克托·迈尔-舍恩伯格与肯尼斯·库克耶合著，讨论了数据科学的应用..."
    chunks = rag.split_text(text)
    
    # 2. 实体识别（两种模式可选）
    entities = rag.extract_entities_hf(text)
    # entities = rag.extract_entities_zhipu(text)
    
    # 3. 关系抽取
    relationships = []
    for chunk in chunks:
        rel_type = rag.extract_relationships(chunk, entities)
        relationships.append({
            "source": "e1",
            "target": "e2", 
            "type": rel_type,
            "weight": 0.9,
            "text_unit_ids": [0,1]
        })
    
    # 4. 创建图节点
    sample_entities = [{
        "id": "e1",
        "title": "大数据时代",
        "type": "书籍",
        "description": "关于大数据应用的权威著作"
    }]
    rag.create_neo4j_nodes(sample_entities)
    
    # 5. 创建关系
    rag.create_neo4j_relationships(relationships)
