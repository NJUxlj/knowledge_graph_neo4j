import re
import json
from py2neo import Graph
from collections import defaultdict

'''
读取三元组，并将数据写入neo4j
'''



graph = Graph("http://localhost:7474", auth=("neo4j", "123456"))

attribute_data = defaultdict(dict)
relation_data = defaultdict(dict)

label_data = {}


# 有的实体后面有括号，里面的内容可以作为标签
# 提取到标签后，把括号部分删除
def get_label_then_clean(x, label_data):
    '''
    ## 作用：
        1. 提取实体x的标签
        2. 把标签从实体中删除
        3. 将 <实体，标签> 存入 label_data 字典中
        4. 返回处理后的实体x
    x: str: 实体
    label_data: dict: 存储实体-标签的字典 【全局字典】
    
    使用正则表达式 （.+） 匹配中文括号及其包含的内容
        返回第一个成功匹配的完整字符串（包含中文括号本身）
        例如当 x = "七里香（歌曲）" 时，返回的是完整的 （歌曲）
    '''
    if re.search("（.+）", x):   
        label_string = re.search('（.+）', x).group(group=0)    # group: 获取默认是第0个捕获组的内容，如果不用group就会返回一个match object
        for label in ["歌曲", "专辑", "电影", "电视剧"]:
            if label in label_string:
                x = re.sub("（.+）", "", x)   # 括号内的内容删掉，因为括号是特殊字符会影响cypher语句运行
                label_data[x] = label
            else:
                x = re.sub("（.+）", "", x)   # 抽取到的标签不是我们想要的，直接删掉就行
    return x
        
        
        
#读取实体-关系-实体三元组文件
with open("triplets_head_rel_tail.txt", encoding="utf8") as f:
    for line in f:
        head, relation, tail = line.strip().split("\t")  # 取出三元组
        head = get_label_then_clean(head, label_data)
        relation_data[head][relation] = tail
        
        
#读取实体-属性-属性值三元组文件
with open("triplets_enti_attr_value.txt", encoding="utf8") as f:
    for line in f:
        pass
    
    



#构建cypher语句
cypher = ""