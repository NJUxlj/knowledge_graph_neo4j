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
        
        

    def load_question_templet(self, templet_path):
        '''
        加载模板信息
        '''
        dataframe = pandas.read_excel(templet_path)
        self.question_templet = []
        
        for index in range(len(dataframe)):
            question = dataframe[question][index]
            cypher = dataframe[cypher][index]
            cypher_check = dataframe[cypher_check][index]
            answer = dataframe[answer][index]
            
            self.question_templet.append([
                question, cypher, cypher_check, answer
            ])
             
        return
    
    
    
    def get_mention_entitys(self, sentence):
        '''
        #获取问题中谈到的实体，可以使用基于词表的方式，也可以使用NER模型
        '''
        return re.findall("|".join(self.entity_set), sentence)
    
    
    def get_mention_relations(self,sentence):
        '''
        # 获取问题中谈到的关系，也可以使用各种文本分类模型
        '''
        return re.findall("|".join(self.relation_set), sentence)
    
    
    
    def get_mention_attributes(self, sentence):
        return re.findall(pattern="|".join(self.attribute_set), string=sentence)
    
    
    
    def get_mention_labels(self, sentence):
        return re.findall(pattern="|".join(self.label_set), string=sentence)
    
    
    
    
    def parse_sentence(self, sentence)->Dict:
        '''
        ## Function:
        - 对sentence进行预处理，提取所给句子中需要的信息 <实体，关系，属性，标签>， 每个都是一个列表。
        - 将所有抽取出的信息列表包装成一个子列表进行返回
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
    
    
    
    def decode_value_combination(self, value_combination, cypher_check):
        '''
        将提取的值分配到键上
        
        index: 当前槽位的索引，用于从value_combination中获取对应的值
        key: 占位符名称，如%ENT%、%REL%
        required_count: 该占位符需要的值的数量
        '''
        
        res = {} # 存储 占位符->值 的映射关系

        '''
        index: 当前槽位的索引，用于从value_combination中获取对应的值
        key: 占位符名称，如%ENT%、%REL%
        required_count: 该占位符需要的值的数量
        '''
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count==1:
                res[key] = value_combination[index][0]
            else:
                '''
                当需要多个值时，为每个值生成带编号的占位符
                    例如：%ENT%变为%ENT0%和%ENT1%
                    生成结果如：{"%ENT0%":"周杰伦", "%ENT1%":"方文山"}
                '''
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
                
        return res
            
    
    
    def get_combinations(self, cypher_check, info):
        '''
        #对于找到了超过模板中需求的实体数量的情况，需要进行排列组合
        #info:{"%ENT%":["周杰伦", "方文山"], “%REL%”:[“作曲”]}
        
        该方法的主要作用是根据模板中定义的槽位需求(cypher_check)，
            对从问题中提取的信息(info)进行排列组合，生成所有可能的组合方式。

        
        ## Args:
        - cypher_check: 模板中定义的槽位需求，格式如{"%ENT%":2, "%REL%":1}，表示需要2个实体和1个关系
        - info: 从问题中提取的信息，格式如{"%ENT%":["周杰伦","方文山"], "%REL%":["作曲"]}
        '''
        slot_values = []
        # 遍历每个槽位需求
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))  # 对每个槽位(%ENT%, %REL%等)，从info中获取对应的值列表
            
        value_combinations = itertools.product(*slot_values) # 对每个槽位的combinations做累乘， 
        '''
        使用itertools.product计算所有槽位组合的笛卡尔积
            例如：如果有两个槽位，一个槽位有2种组合，另一个有1种，则总共生成2×1=2种组合
        '''

        combinations = []
        '''
        对每种组合调用decode_value_combination方法进行解码
            解码后的格式如：{"%ENT1%":"周杰伦", "%ENT2%":"方文山", "%REL%":"作曲"}
        '''
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
            
        return  combinations
    
    
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
    
    