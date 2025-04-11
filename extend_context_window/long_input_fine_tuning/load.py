import torch

import os
import json
import random
import numpy as np
import logging
import pandas as pd  
import requests  
import zipfile  
import io  
from tqdm import tqdm  
from datasets import load_dataset, Dataset  
from typing import Dict, List, Optional, Union, Any  





def load_loogle_dataset(task_type: str = "longqa") -> Dataset:  
    """  
    加载LooGLE数据集  
    
    Args:  
        task_type: 任务类型, "longqa"(长文档QA)或"shortqa"(短文档QA)  
    
    Returns:  
        包含文档、问题和答案的数据集  
    """  
    # LooGLE数据集可能需要从官方仓库下载  
    # 这里提供一个基于论文描述的模拟实现  
    # 实际使用时应替换为实际数据集的加载代码  
    
    base_url = "https://github.com/psunlpgroup/LooGLE/raw/main/data/"  
    
    # 检查本地缓存  
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "lift_evaluation")  
    os.makedirs(cache_dir, exist_ok=True)  
    
    local_file = os.path.join(cache_dir, f"loogle_{task_type}.json")  
    
    # 如果本地没有缓存，则下载  
    if not os.path.exists(local_file):  
        print(f"下载LooGLE {task_type}数据集...")  
        try:  
            response = requests.get(f"{base_url}{task_type}.json")  
            if response.status_code == 200:  
                with open(local_file, "wb") as f:  
                    f.write(response.content)  
            else:  
                # 如果下载失败，创建一个模拟数据集  
                print(f"无法下载LooGLE数据集，创建模拟数据...")  
                create_mock_loogle_dataset(local_file, task_type)  
        except:  
            # 如果下载出错，创建一个模拟数据集  
            print(f"下载LooGLE数据集出错，创建模拟数据...")  
            create_mock_loogle_dataset(local_file, task_type)  
    
    # 加载数据集  
    with open(local_file, "r", encoding="utf-8") as f:  
        raw_data = json.load(f)  
    
    # 转换为Datasets格式  
    if isinstance(raw_data, list):  
        dataset = Dataset.from_dict({  
            "document": [item.get("document", "") for item in raw_data],  
            "question": [item.get("question", "") for item in raw_data],  
            "answer": [item.get("answer", "") for item in raw_data]  
        })  
    else:  
        # 如果格式不是列表，尝试不同的解析方法  
        dataset = Dataset.from_dict({  
            "document": [raw_data[i].get("document", "") for i in raw_data],  
            "question": [raw_data[i].get("question", "") for i in raw_data],  
            "answer": [raw_data[i].get("answer", "") for i in raw_data]  
        })  
    
    return dataset  

def create_mock_loogle_dataset(output_file: str, task_type: str):  
    """创建模拟LooGLE数据集"""  
    # 基于论文描述创建模拟数据  
    mock_data = []  
    
    # 生成不同大小的模拟数据  
    num_samples = 20  
    
    for i in range(num_samples):  
        # 文档长度取决于任务类型  
        doc_length = 20000 if task_type == "longqa" else 2000  
        
        mock_data.append({  
            "document": f"这是一个{'长' if task_type == 'longqa' else '短'}文本示例，用于测试LIFT方法。"  
                        f"文档编号:{i} " + " ".join([f"内容{j}" for j in range(doc_length)]),  
            "question": f"这个文档的编号是多少？",  
            "answer": f"{i}"  
        })  
    
    # 保存到文件  
    with open(output_file, "w", encoding="utf-8") as f:  
        json.dump(mock_data, f, ensure_ascii=False, indent=2)  

def load_longbench_dataset(task_name: str) -> Dataset:  
    """  
    加载LongBench数据集  
    
    Args:  
        task_name: 任务名称, 如"summarization", "qa", "few_shot_learning"等  
    
    Returns:  
        包含文档、问题和答案的数据集  
    """  
    # LongBench是一个长文本评测基准  
    # 可以从https://github.com/THUDM/LongBench获取  
    
    # 检查本地缓存  
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "lift_evaluation")  
    os.makedirs(cache_dir, exist_ok=True)  
    
    local_file = os.path.join(cache_dir, f"longbench_{task_name}.json")  
    
    # 如果本地没有缓存，则下载或创建模拟数据  
    if not os.path.exists(local_file):  
        print(f"创建LongBench {task_name}模拟数据集...")  
        create_mock_longbench_dataset(local_file, task_name)  
    
    # 加载数据集  
    with open(local_file, "r", encoding="utf-8") as f:  
        raw_data = json.load(f)  
    
    # 转换为Datasets格式，根据不同任务调整字段  
    if task_name == "summarization":  
        dataset = Dataset.from_dict({  
            "document": [item.get("input", "") for item in raw_data],  
            "question": ["总结以下文档内容:" for _ in raw_data],  
            "answer": [item.get("output", "") for item in raw_data]  
        })  
    elif task_name == "qa":  
        dataset = Dataset.from_dict({  
            "document": [item.get("context", "") for item in raw_data],  
            "question": [item.get("question", "") for item in raw_data],  
            "answer": [item.get("answer", "") for item in raw_data]  
        })  
    elif task_name == "few_shot_learning":  
        # 处理小样本学习任务  
        examples = [" ".join([f"示例{i}: 问题: {e['question']} 答案: {e['answer']}"   
                             for i, e in enumerate(item.get("examples", []))])   
                   for item in raw_data]  
        
        dataset = Dataset.from_dict({  
            "document": examples,  
            "question": [item.get("question", "") for item in raw_data],  
            "answer": [item.get("answer", "") for item in raw_data]  
        })  
    elif task_name == "code_completion":  
        dataset = Dataset.from_dict({  
            "document": [item.get("context", "") for item in raw_data],  
            "question": ["请完成以下代码:" for _ in raw_data],  
            "answer": [item.get("completion", "") for item in raw_data]  
        })  
    else:  # synthetic_tasks等其他任务  
        dataset = Dataset.from_dict({  
            "document": [item.get("context", "") for item in raw_data],  
            "question": [item.get("question", "") for item in raw_data],  
            "answer": [item.get("answer", "") for item in raw_data]  
        })  
    
    return dataset  

def create_mock_longbench_dataset(output_file: str, task_name: str):  
    """创建模拟LongBench数据集"""  
    mock_data = []  
    num_samples = 20  
    
    # 根据任务类型创建不同的模拟数据  
    if task_name == "summarization":  
        for i in range(num_samples):  
            long_text = f"这是一个需要总结的长文档 {i}。" + " ".join([f"内容{j}" for j in range(5000)])  
            mock_data.append({  
                "input": long_text,  
                "output": f"这是文档{i}的摘要，主要讨论了内容0到内容4999。"  
            })  
    
    elif task_name == "qa":  
        for i in range(num_samples):  
            context = f"这是一个问答任务的上下文 {i}。" + " ".join([f"事实{j}：值{j}" for j in range(500)])  
            mock_data.append({  
                "context": context,  
                "question": f"事实{i*10}的值是多少？",  
                "answer": f"值{i*10}"  
            })  
    
    elif task_name == "few_shot_learning":  
        for i in range(num_samples):  
            # 创建示例  
            examples = []  
            for j in range(5):  # 5个示例  
                examples.append({  
                    "question": f"示例问题{j}是什么?",  
                    "answer": f"示例答案{j}"  
                })  
            
            mock_data.append({  
                "examples": examples,  
                "question": f"根据以上示例，问题{i}的答案是什么?",  
                "answer": f"答案{i}"  
            })  
    
    elif task_name == "code_completion":  
        for i in range(num_samples):  
            context = f"""  
                        def factorial(n):  
                            \"\"\"计算阶乘 {i}\"\"\"  
                            if n == 0:  
                                return 1  
                            else:  
                                return n *   
                                    """  
            mock_data.append({  
                "context": context,  
                "completion": f"factorial(n-1)  # 完成阶乘函数 {i}"  
            })  
    
    else:  # synthetic_tasks  
        for i in range(num_samples):  
            mock_data.append({  
                "context": f"这是一个合成任务{task_name}的长上下文 {i}。" + " ".join([f"内容{j}" for j in range(2000)]),  
                "question": f"在上下文中找到关键信息{i}",  
                "answer": f"关键信息{i}在内容{i*100}中"  
            })  
    
    # 保存到文件  
    with open(output_file, "w", encoding="utf-8") as f:  
        json.dump(mock_data, f, ensure_ascii=False, indent=2)  

def load_custom_dataset(file_path: str) -> Dataset:  
    """  
    加载自定义领域数据集  
    
    Args:  
        file_path: 数据集文件路径，支持.json, .csv, .txt格式  
    
    Returns:  
        包含文档、问题和答案的数据集  
    """  
    extension = os.path.splitext(file_path)[1].lower()  
    
    # 如果文件不存在，创建一个模拟数据集  
    if not os.path.exists(file_path):  
        print(f"文件 {file_path} 不存在，创建模拟自定义数据集...")  
        if not os.path.exists(os.path.dirname(file_path)):  
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  
        
        create_mock_custom_dataset(file_path)  
        
    # 根据文件类型加载数据  
    if extension == '.json':  
        with open(file_path, 'r', encoding='utf-8') as f:  
            data = json.load(f)  
        
        # 尝试解析不同的JSON格式  
        if isinstance(data, list):  
            # 列表格式  
            if all(isinstance(item, dict) for item in data):  
                if all(key in data[0] for key in ['document', 'question', 'answer']):  
                    # 标准格式  
                    dataset = Dataset.from_dict({  
                        "document": [item.get("document", "") for item in data],  
                        "question": [item.get("question", "") for item in data],  
                        "answer": [item.get("answer", "") for item in data]  
                    })  
                else:  
                    # 猜测其他可能的key名  
                    doc_keys = ['document', 'text', 'content', 'context', 'passage']  
                    q_keys = ['question', 'query', 'q']  
                    a_keys = ['answer', 'response', 'a', 'output']  
                    
                    doc_key = next((k for k in doc_keys if k in data[0]), doc_keys[0])  
                    q_key = next((k for k in q_keys if k in data[0]), q_keys[0])  
                    a_key = next((k for k in a_keys if k in data[0]), a_keys[0])  
                    
                    dataset = Dataset.from_dict({  
                        "document": [item.get(doc_key, "") for item in data],  
                        "question": [item.get(q_key, "") for item in data],  
                        "answer": [item.get(a_key, "") for item in data]  
                    })  
            else:  
                # 不是字典列表格式，可能只是文本列表  
                # 创建简单问题："请总结这段文本"  
                dataset = Dataset.from_dict({  
                    "document": data,  
                    "question": ["请总结这段文本的主要内容" for _ in data],  
                    "answer": ["" for _ in data]  # 空答案，评测时需人工判断  
                })  
        else:  
            # 可能是其他格式，尝试转换  
            documents = []  
            questions = []  
            answers = []  
            
            # 遍历所有可能的键  
            for key, value in data.items():  
                if isinstance(value, dict):  
                    if 'text' in value or 'content' in value:  
                        doc = value.get('text', value.get('content', ''))  
                        documents.append(doc)  
                        questions.append(f"关于'{key}'的信息是什么?")  
                        answers.append(doc[:100])  # 取前100个字符作为"答案"  
            
            dataset = Dataset.from_dict({  
                "document": documents,  
                "question": questions,  
                "answer": answers  
            })  
    
    elif extension == '.csv':  
        # 读取CSV文件  
        df = pd.read_csv(file_path)  
        
        # 猜测列名  
        doc_cols = ['document', 'text', 'content', 'context', 'passage']  
        q_cols = ['question', 'query', 'q']  
        a_cols = ['answer', 'response', 'a', 'output']  
        
        doc_col = next((col for col in doc_cols if col in df.columns), None)  
        q_col = next((col for col in q_cols if col in df.columns), None)  
        a_col = next((col for col in a_cols if col in df.columns), None)  
        
        if doc_col and q_col and a_col:  
            # 标准格式  
            dataset = Dataset.from_pandas(df[[doc_col, q_col, a_col]].rename(  
                columns={doc_col: 'document', q_col: 'question', a_col: 'answer'}  
            ))  
        elif doc_col:  
            # 只有文档列，创建简单问题  
            dataset = Dataset.from_dict({  
                "document": df[doc_col].tolist(),  
                "question": ["请总结这段文本的主要内容" for _ in range(len(df))],  
                "answer": ["" for _ in range(len(df))]  
            })  
        else:  
            # 找不到合适的列，使用所有列作为文档  
            # 将每行所有列的内容合并为一个文档  
            documents = [" | ".join([f"{col}: {row[col]}" for col in df.columns])   
                         for _, row in df.iterrows()]  
            
            dataset = Dataset.from_dict({  
                "document": documents,  
                "question": ["请总结这段文本的主要内容" for _ in range(len(df))],  
                "answer": ["" for _ in range(len(df))]  
            })  
    
    elif extension == '.txt':  
        # 读取文本文件  
        with open(file_path, 'r', encoding='utf-8') as f:  
            lines = f.readlines()  
        
        # 每行作为一个文档  
        dataset = Dataset.from_dict({  
            "document": lines,  
            "question": ["请总结这段文本的主要内容" for _ in lines],  
            "answer": ["" for _ in lines]  
        })  
    
    else:  
        # 不支持的文件类型，创建空数据集  
        print(f"不支持的文件类型: {extension}")  
        dataset = Dataset.from_dict({  
            "document": [],  
            "question": [],  
            "answer": []  
        })  
    
    return dataset  

def create_mock_custom_dataset(output_file: str):  
    """创建模拟自定义数据集"""  
    extension = os.path.splitext(output_file)[1].lower()  
    num_samples = 20  
    
    if extension == '.json':  
        mock_data = []  
        for i in range(num_samples):  
            doc_length = 3000 + i * 500  # 文档长度从3000到12500不等  
            mock_data.append({  
                "document": f"这是一个自定义领域的文档 {i}。" + " ".join([f"专业内容{j}" for j in range(doc_length)]),  
                "question": f"在文档{i}中，专业内容{i*100}的描述是什么?",  
                "answer": f"专业内容{i*100}"  
            })  
        
        with open(output_file, "w", encoding="utf-8") as f:  
            json.dump(mock_data, f, ensure_ascii=False, indent=2)  
    
    elif extension == '.csv':  
        import csv  
        with open(output_file, "w", encoding="utf-8", newline='') as f:  
            writer = csv.writer(f)  
            # 写入表头  
            writer.writerow(["document", "question", "answer"])  
            
            # 写入数据  
            for i in range(num_samples):  
                doc_length = 3000 + i * 500  
                document = f"这是一个自定义领域的文档 {i}。" + " ".join([f"专业内容{j}" for j in range(doc_length)])  
                question = f"在文档{i}中，专业内容{i*100}的描述是什么?"  
                answer = f"专业内容{i*100}"  
                writer.writerow([document, question, answer])  
    
    elif extension == '.txt':  
        with open(output_file, "w", encoding="utf-8") as f:  
            for i in range(num_samples):  
                doc_length = 100 + i * 50  # 文本文件中每行长度适中  
                f.write(f"这是一个自定义领域的文档 {i}。" + " ".join([f"专业内容{j}" for j in range(doc_length)]) + "\n")  

def load_training_long_text(file_path: Optional[str] = None) -> str:  
    """  
    加载用于训练LIFT的长文本  
    
    Args:  
        file_path: 长文本文件路径，如果为None则使用模拟数据  
    
    Returns:  
        长文本字符串  
    """  
    if file_path and os.path.exists(file_path):  
        # 从文件加载长文本  
        with open(file_path, 'r', encoding='utf-8') as f:  
            return f.read()  
    else:  
        # 创建模拟长文本  
        print("创建模拟长训练文本...")  
        paragraphs = []  
        # 生成约10万词的文本  
        for i in range(100):  
            para_length = 1000  # 每段约1000词  
            paragraph = f"第{i+1}段：这是一个用于训练LIFT模型的长文本段落。" + " ".join([f"内容{i}_{j}" for j in range(para_length)])  
            paragraphs.append(paragraph)  
        
        return "\n\n".join(paragraphs)  





def prepare_evaluation_datasets():  
    """准备评测数据集"""  
    datasets = {  
        # LooGLE基准 (论文中重点使用的评测集)  
        "loogle": {  
            "longqa": load_loogle_dataset("longqa"),  # 长文档QA任务  
            "shortqa": load_loogle_dataset("shortqa")  # 短文档QA任务  
        },  
        
        # LongBench基准 (选择代表性任务)  
        "longbench": {  
            "summarization": load_longbench_dataset("summarization"),  
            "qa": load_longbench_dataset("qa"),  
            "few_shot_learning": load_longbench_dataset("few_shot_learning"),  
            "code_completion": load_longbench_dataset("code_completion"),  
            "synthetic_tasks": load_longbench_dataset("synthetic_tasks")  
        },  
        
        # 自定义评测集 (针对特定领域)  
        "custom": load_custom_dataset("/path/to/your/domain/data")  
    }  
    return datasets  




class EvaluationDataLoader:  
    """评测数据加载器"""  
    def __init__(self, dataset, tokenizer, max_length=4096):  
        self.dataset = dataset  
        self.tokenizer = tokenizer  
        self.max_length = max_length  
        
    def __iter__(self):  
        for example in self.dataset:  
            # 处理文档和问题  
            document = example["document"]  
            question = example["question"]  
            ground_truth = example["answer"]  
            
            # 构建不同评测场景  
            # 1. LIFT模式: 只有问题，依赖模型参数记忆  
            lift_input = question  
            
            # 2. ICL模式: 完整上下文+问题 (如果适合上下文窗口)  
            icl_input = document + "\n\n" + question  
            
            # 3. 截断ICL模式: 截断上下文+问题 (对超长文档)  
            truncated_doc = self.truncate_document(document, max_length=self.max_length-len(question)-10)  
            truncated_icl_input = truncated_doc + "\n\n" + question  
            
            yield {  
                "lift_input": lift_input,  
                "icl_input": icl_input,  
                "truncated_icl_input": truncated_icl_input,  
                "ground_truth": ground_truth,  
                "document": document,  
                "question": question  
            }  
    
    def truncate_document(self, document, max_length):  
        """智能截断文档，保留开头和结尾部分"""  
        doc_tokens = self.tokenizer.encode(document)  
        if len(doc_tokens) <= max_length:  
            return document  
            
        # 保留开头和结尾  
        prefix_len = max_length // 2  
        suffix_len = max_length - prefix_len  
        
        prefix = self.tokenizer.decode(doc_tokens[:prefix_len])  
        suffix = self.tokenizer.decode(doc_tokens[-suffix_len:])  
        
        return prefix + "\n...[内容省略]...\n" + suffix  
    
    
    
    
    

def prepare_lift_evaluation():  
    """准备LIFT评测环境"""  
    # 1. 设置缓存目录  
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "lift_evaluation")  
    os.makedirs(cache_dir, exist_ok=True)  
    
    # 2. 加载评测数据集  
    print("加载评测数据集...")  
    datasets = prepare_evaluation_datasets()  
    
    # 3. 打印数据集统计信息  
    for benchmark, tasks in datasets.items():  
        print(f"\n{benchmark.upper()}基准:")  
        if isinstance(tasks, dict):  
            for task_name, dataset in tasks.items():  
                print(f"  - {task_name}: {len(dataset)}个样本")  
                # 打印示例  
                if len(dataset) > 0:  
                    sample = dataset[0]  
                    print(f"    示例文档: {sample['document'][:100]}...")  
                    print(f"    示例问题: {sample['question']}")  
                    print(f"    示例答案: {sample['answer']}")  
        else:  
            print(f"  - 样本数: {len(tasks)}")  
    
    # 4. 准备训练用长文本  
    training_text = load_training_long_text()  
    print(f"\n训练用长文本: {len(training_text)}字符")  
    print(f"前100字符预览: {training_text[:100]}...")  
    
    return {  
        "datasets": datasets,  
        "training_text": training_text  
    }  








if __name__ == '__main__':
    evaluation_resources = prepare_lift_evaluation()  
    print("\n评测资源准备完成!")  