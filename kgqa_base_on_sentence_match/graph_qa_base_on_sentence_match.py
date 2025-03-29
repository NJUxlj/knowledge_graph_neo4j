import re
import json
import pandas
import itertools
from py2neo import Graph
from collections import defaultdict

from typing import Dict, List, Tuple, Union, Literal, Optional
'''
使用文本匹配的方式进行知识图谱的使用
'''





class GraphQA:
    def __init__(self, ):
        graph = Graph()
        
        schema_path = None
        
        templet_path = None
        
        self.load(schema_path, templet_path)
        
        print("知识图谱问答系统加载完毕！\n===============")
        
        
        
    def load(self, schema_path, templet_path):
        '''
        加载知识图谱的schema和模板
        '''
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)
        return
    
    def load_kg_schema(self, path):
        with open(path, encoding = 'utf8') as f:
            schema = json.load(f)
        self.relation_set  = set(schema["relations"])
        self.entity_set = set(schema['entitys'])
        self.label_set = set(schema['labels'])
        self.attribute_set = set(schema['attributes'])
        
        return
        
        

    def load_question_templet(self):
        pass
    
    
    
    def get_mention_entitys(self, sentence):
        pass
    
    
    def get_mention_relations(self,sentence):
        pass
    
    
    
    def get_mention_attributes(self, sentence):
        pass
    
    
    
    def get_mention_labels(self, sentence):
        pass
    
    
    
    
    def parse_sentence(self, sentence)->Dict:
        '''
        ## Function:
        - 对sentence进行预处理，提取所给句子中需要的信息 <实体，关系，属性，标签>， 每个都是一个列表。
        - 将所有抽取出的信息列表包装成一个子带你进行返回
        - 
        '''
        
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        attributes = self.get_mention_attributes(sentence)
        labels = self.get_mention_labels(sentence)
        
        return {
            "%ENT%": entitys,
            "%REL%": relations,
            "%LAB%": labels,
            "%ATT%": attributes,
        }
    
    
    
    def decode_value_combination(self, ):
        pass
    
    
    def get_combinations(self):
        pass
    
    
    def replace_token_in_string(self, string:str, combinations:Dict):
        '''
        #将带有token的模板替换成真实词
        #string: %ENT1%和%ENT2%是%REL%关系吗
        #combination: {"%ENT1%":"word1", "%ENT2%":"word2", "%REL%":"word"}  # 填充词到真实词的映射字典
        '''
        for key,value in combinations.items():
            string.replace(key, value)
            
        return string
    
    
    def parse_result(self, graph_search_result, answer, info):
        '''
         解析知识图谱的查询结果
        '''
        graph_search_result = graph_search_result[0]
        
        # 关系查找返回的结果形式较为特殊，单独处理
        if "REL" in graph_search_result:
            pass
        
        
        answer = self.replace_token_in_string(answer, graph_search_result)

        return answer
    
    def query(self, sentence):
        print("============")
        print("用户的原始输入是：",sentence)
        
        info:Dict = self.parse_sentence(sentence) # 将句子中的所有元素抽取成一个Dict[List]

        templet_cypher_score = self.cypher_match(sentence, info)  # cypher 匹配


        # 遍历所有的模板 和 对应填充后的cypher语句
        for templet, cypher, score, answer in templet_cypher_score:
            graph_search_result = self.graph.run(cypher).data()
    
            # 最高分命中的模板不一定在图上能找到答案, 当不能找到答案时，运行下一个搜索语句, 找到答案时停止查找后面的模板
        
            if graph_search_result:
                answer = self.parse_result(graph_search_result, answer, info)
                return answer
            
        return None
        
    
    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    graph = GraphQA()
    res = graph.query("谁导演的不能说的秘密")
    print(res)
    
    