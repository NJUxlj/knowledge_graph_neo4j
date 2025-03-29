import re
import json
import pandas
import itertools
from py2neo import Graph
from collections import defaultdict


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
        self.load_kg_schema()
        self.load_question_templet()
        return
    
    

    
    def load_kg_schema(self):
        pass
    
    
    
    def load_question_templet(self):
        pass
    
    
    
    def get_mention_entitys(self, sentence):
        pass
    