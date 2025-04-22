import json  
import matplotlib.pyplot as plt  
import numpy as np  
import collections  
from tqdm import tqdm  


import torch


from transformers import (
   AutoTokenizer,
   AutoModelForCausalLM,
   AutoModelForSequenceClassification, 
)


from lift_tuning import LIFTConfig, LIFT

from .evaluate import LIFTEvaluator
from .load import (
    prepare_evaluation_datasets, 
    EvaluationDataLoader,
    
)


def evaluate_lift_model():  
    """全面评测LIFT模型"""  
    # 配置  
    config = LIFTConfig(  
        base_model_name="meta-llama/Llama-3-8B-Instruct",  
        max_length=4096,  
        segment_length=2048,  
        overlap_length=512,  
        use_gated_memory=True,  
        use_contextualized_training=True  
    )  
    
    # 1. 加载模型  
    print("加载模型...")  
    # LIFT微调模型  
    lift = LIFT(config)  
    lift.model.model = AutoModelForCausalLM.from_pretrained("./lift_model")  
    lift.tokenizer = AutoTokenizer.from_pretrained("./lift_model")  
    
    # 原始基础模型 (用于ICL比较)  
    base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)  
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)  
    
    # 2. 包装基础模型为评测用的生成器  
    class BaseModelGenerator:  
        def __init__(self, model, tokenizer, max_new_tokens=100):  
            self.model = model  
            self.tokenizer = tokenizer  
            self.max_new_tokens = max_new_tokens  
            
        def generate_answer(self, prompt):  
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)  
            with torch.no_grad():  
                outputs = self.model.generate(  
                    inputs["input_ids"],  
                    max_new_tokens=self.max_new_tokens,  
                    temperature=0.7,  
                    num_return_sequences=1  
                )  
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
            # 只返回新生成的部分  
            answer = generated_text[len(prompt):].strip()  
            return answer  
    
    base_generator = BaseModelGenerator(base_model, base_tokenizer)  
    
    # 3. 准备评测器  
    evaluator = LIFTEvaluator(lift, base_generator)  
    
    # 4. 加载评测数据集  
    print("加载评测数据集...")  
    datasets = prepare_evaluation_datasets()  
    
    # 5. 运行评测  
    all_results = {}  
    
    # 5.1 评测LooGLE数据集  
    print("评测LooGLE数据集...")  
    loogle_results = {}  
    for task_name, dataset in datasets["loogle"].items():  
        print(f"评测任务: {task_name}")  
        data_loader = EvaluationDataLoader(dataset, lift.tokenizer)  
        results = evaluator.evaluate_dataset(data_loader)  
        loogle_results[task_name] = results  
    all_results["loogle"] = loogle_results  
    
    # 5.2 评测LongBench数据集  
    print("评测LongBench数据集...")  
    longbench_results = {}  
    for task_name, dataset in datasets["longbench"].items():  
        print(f"评测任务: {task_name}")  
        data_loader = EvaluationDataLoader(dataset, lift.tokenizer)  
        results = evaluator.evaluate_dataset(data_loader)  
        longbench_results[task_name] = results  
    all_results["longbench"] = longbench_results  
    
    # 5.3 评测自定义数据集  
    print("评测自定义数据集...")  
    custom_data_loader = EvaluationDataLoader(datasets["custom"], lift.tokenizer)  
    custom_results = evaluator.evaluate_dataset(custom_data_loader)  
    all_results["custom"] = custom_results  
    
    # 6. 保存结果  
    with open("lift_evaluation_results.json", "w") as f:  
        json.dump(all_results, f, indent=2)  
    
    # 7. 可视化结果  
    visualize_results(all_results)  
    
    return all_results  

def visualize_results(results):  
    """可视化评测结果"""  
    # 提取LooGLE结果进行可视化  
    loogle_results = results.get("loogle", {})  
    
    # 创建图表  
    plt.figure(figsize=(12, 8))  
    
    # 设置任务和方法  
    tasks = list(loogle_results.keys())  
    methods = ["LIFT", "ICL", "Truncated ICL"]  
    metrics = ["exact_match", "f1_score", "rouge_l"]  
    colors = ["blue", "green", "red"]  
    
    # 设置x轴位置  
    x = np.arange(len(tasks))  
    width = 0.25  # 柱状图宽度  
    
    # 绘制f1_score柱状图  
    for i, method in enumerate(methods):  
        method_values = []  
        for task in tasks:  
            if method == "LIFT":  
                task_results = loogle_results[task]["lift"]  
            elif method == "ICL":  
                task_results = loogle_results[task]["icl"]  
            else:  
                task_results = loogle_results[task]["truncated_icl"]  
                
            # 如果结果不存在，使用0  
            if task_results:  
                method_values.append(task_results["f1_score"])  
            else:  
                method_values.append(0)  
                
        plt.bar(x + width*(i-1), method_values, width, label=method, color=colors[i])  
    
    # 添加标签和标题  
    plt.xlabel('Tasks')  
    plt.ylabel('F1 Score')  
    plt.title('LIFT vs ICL Performance on LooGLE Tasks')  
    plt.xticks(x, tasks)  
    plt.legend()  
    
    # 添加网格线  
    plt.grid(axis='y', linestyle='--', alpha=0.7)  
    
    # 保存图表  
    plt.tight_layout()  
    plt.savefig("lift_evaluation_results.png")  
    plt.close()  
    
    # 更多可视化...  
    
    
    
    
    
    
def run_controlled_experiments():  
    """运行控制变量实验"""  
    # 基础配置  
    base_config = LIFTConfig(  
        base_model_name="meta-llama/Llama-3-8B-Instruct",  
        max_length=4096,  
        segment_length=2048,  
        overlap_length=512  
    )  
    
    # 实验配置变量  
    experiments = [  
        # 1. 基础LIFT (无门控记忆、无上下文化训练)  
        {"name": "base_lift", "config": {**base_config.__dict__, "use_gated_memory": False, "use_contextualized_training": False}},  
        
        # 2. LIFT + 门控记忆  
        {"name": "lift_with_gm", "config": {**base_config.__dict__, "use_gated_memory": True, "use_contextualized_training": False}},  
        
        # 3. LIFT + 上下文化训练  
        {"name": "lift_with_ct", "config": {**base_config.__dict__, "use_gated_memory": False, "use_contextualized_training": True}},  
        
        # 4. 完整LIFT (门控记忆 + 上下文化训练)  
        {"name": "full_lift", "config": {**base_config.__dict__, "use_gated_memory": True, "use_contextualized_training": True}},  
        
        # 5. 不同分段长度的LIFT  
        {"name": "lift_small_seg", "config": {**base_config.__dict__, "segment_length": 1024, "overlap_length": 256}},  
        {"name": "lift_large_seg", "config": {**base_config.__dict__, "segment_length": 3072, "overlap_length": 768}},  
        
        # 6. 不同QA对数量的LIFT  
        {"name": "lift_few_qa", "config": {**base_config.__dict__, "num_qa_pairs": 5}},  
        {"name": "lift_many_qa", "config": {**base_config.__dict__, "num_qa_pairs": 20}}  
    ]  
    
    # 选择评测数据集(简化版)  
    evaluation_datasets = {  
        "loogle_longqa": load_loogle_dataset("longqa"),  
        "longbench_qa": load_longbench_dataset("qa")  
    }  
    
    # 运行实验  
    experiment_results = {}  
    for exp in experiments:  
        print(f"运行实验: {exp['name']}")  
        
        # 创建配置  
        config = LIFTConfig(**exp["config"])  
        
        # 训练LIFT模型  
        lift = LIFT(config)  
        long_text = load_training_long_text()  # 加载训练用长文本  
        lift.train(long_text, num_epochs=3, batch_size=4)  
        
        # 评测模型  
        results = {}  
        for dataset_name, dataset in evaluation_datasets.items():  
            print(f"  评测数据集: {dataset_name}")  
            data_loader = EvaluationDataLoader(dataset, lift.tokenizer)  
            evaluator = LIFTEvaluator(lift)  
            results[dataset_name] = evaluator.evaluate_dataset(data_loader)  
        
        experiment_results[exp["name"]] = results  
    
    # 保存结果  
    with open("lift_controlled_experiments.json", "w") as f:  
        json.dump(experiment_results, f, indent=2)  
    
    # 可视化结果  
    visualize_experiment_results(experiment_results)  
    
    return experiment_results  





def test_custom_document_memory():  
    """测试自定义文档的记忆能力"""  
    # 创建LIFT配置  
    config = LIFTConfig(  
        base_model_name="meta-llama/Llama-3-8B-Instruct",  
        max_length=4096,  
        segment_length=2048,  
        overlap_length=512,  
        use_gated_memory=True,  
        use_contextualized_training=True  
    )  
    
    # 初始化LIFT  
    lift = LIFT(config)  
    
    # 加载自定义文档（例如公司知识库、产品手册等）  
    document = load_document("path/to/your/document.txt")  
    
    # 微调模型以记忆文档内容  
    lift.train(document, num_epochs=3, batch_size=4)  
    
    # 设计测试问题（从简单到复杂）  
    test_questions = [  
        # 1. 直接提取  
        "文档中提到的主要产品是什么？",  
        
        # 2. 事实查找  
        "根据文档，公司成立于哪一年？",  
        
        # 3. 关系理解  
        "文档中提到的A和B有什么关系？",  
        
        # 4. 推理题  
        "基于文档内容，可以得出什么结论？",  
        
        # 5. 综合题  
        "请总结文档的三个主要观点并分析它们之间的联系。"  
    ]  
    
    # 测试并记录结果  
    results = []  
    for question in test_questions:  
        # 使用LIFT模型回答（不提供上下文）  
        lift_answer = lift.answer_question(question)  
        
        # 使用原始模型回答（提供完整文档作为上下文）  
        base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)  
        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)  
        
        prompt = document + "\n\n" + question  
        inputs = base_tokenizer(prompt, return_tensors="pt").to(base_model.device)  
        with torch.no_grad():  
            outputs = base_model.generate(  
                inputs["input_ids"],  
                max_new_tokens=100  
            )  
        base_answer = base_tokenizer.decode(outputs[0])[len(prompt):].strip()  
        
        # 人工评分（后续可由评估者填写）  
        human_score_lift = None  # 1-10分  
        human_score_base = None  # 1-10分  
        
        results.append({  
            "question": question,  
            "lift_answer": lift_answer,  
            "base_answer": base_answer,  
            "human_score_lift": human_score_lift,  
            "human_score_base": human_score_base,  
            "notes": ""  # 评估者笔记  
        })  
    
    # 保存结果  
    with open("custom_document_test_results.json", "w", encoding="utf-8") as f:  
        json.dump(results, f, ensure_ascii=False, indent=2)  
    
    return results  




def test_memory_persistence():  
    """测试记忆持久性"""  
    # 创建LIFT配置  
    config = LIFTConfig(  
        base_model_name="meta-llama/Llama-3-8B-Instruct",  
        max_length=4096,  
        segment_length=2048,  
        overlap_length=512,  
        use_gated_memory=True,  
        use_contextualized_training=True  
    )  
    
    # 初始化LIFT  
    lift = LIFT(config)  
    
    # 1. 首先微调一个长文档A  
    document_a = load_document("document_a.txt")  
    lift.train(document_a, num_epochs=3, batch_size=4)  
    
    # 保存问题和预期答案  
    questions_a = [  
        "文档A的主要内容是什么？",  
        "文档A中提到的关键数据是什么？"  
    ]  
    
    # 测试模型对文档A的记忆  
    results_a1 = []  
    for question in questions_a:  
        answer = lift.answer_question(question)  
        results_a1.append({"question": question, "answer": answer})  
    
    # 2. 然后微调另一个长文档B  
    document_b = load_document("document_b.txt")  
    lift.train(document_b, num_epochs=3, batch_size=4)  
    
    # 保存问题和预期答案  
    questions_b = [  
        "文档B的主要内容是什么？",  
        "文档B中提到的关键数据是什么？"  
    ]  
    
    # 测试模型对文档B的记忆  
    results_b = []  
    for question in questions_b:  
        answer = lift.answer_question(question)  
        results_b.append({"question": question, "answer": answer})  
    
    # 3. 再次测试模型对文档A的记忆  
    results_a2 = []  
    for question in questions_a:  
        answer = lift.answer_question(question)  
        results_a2.append({"question": question, "answer": answer})  
    
    # 4. 比较两次对文档A的回答，评估记忆持久性  
    memory_persistence = []  
    for i in range(len(questions_a)):  
        similarity = compute_answer_similarity(results_a1[i]["answer"], results_a2[i]["answer"])  
        memory_persistence.append({  
            "question": questions_a[i],  
            "answer_before_b": results_a1[i]["answer"],  
            "answer_after_b": results_a2[i]["answer"],  
            "similarity": similarity  
        })  
    
    return {  
        "document_a_before": results_a1,  
        "document_b": results_b,  
        "document_a_after": results_a2,  
        "memory_persistence": memory_persistence  
    }  

def compute_answer_similarity(answer1, answer2):  
    """计算两个答案的相似度"""  
    # 使用简单的词袋模型和余弦相似度  
    from sklearn.feature_extraction.text import CountVectorizer  
    from sklearn.metrics.pairwise import cosine_similarity  
    
    vectorizer = CountVectorizer().fit([answer1, answer2])  
    vectors = vectorizer.transform([answer1, answer2])  
    return cosine_similarity(vectors)[0][1]  




def test_model_scalability():  
    """测试模型规模扩展性"""  
    # 准备不同规模的模型  
    models = [  
        {"name": "Llama-3-8B", "path": "meta-llama/Llama-3-8B-Instruct"},  
        {"name": "Gemma-7B", "path": "google/gemma-7b-it"},  
        {"name": "Llama-2-13B", "path": "meta-llama/Llama-2-13b-chat-hf"},  
        # 更大模型需要更多资源  
    ]  
    
    # 准备评测数据集  
    eval_dataset = load_loogle_dataset("longqa")  
    
    # 针对每个模型运行LIFT并评测  
    results = {}  
    for model_info in models:  
        print(f"测试模型: {model_info['name']}")  
        
        # 创建LIFT配置  
        config = LIFTConfig(  
            base_model_name=model_info["path"],  
            max_length=4096,  
            segment_length=2048,  
            overlap_length=512,  
            use_gated_memory=True,  
            use_contextualized_training=True  
        )  
        
        # 初始化并训练LIFT  
        lift = LIFT(config)  
        long_text = load_training_long_text()  
        lift.train(long_text, num_epochs=3, batch_size=4)  
        
        # 评测  
        data_loader = EvaluationDataLoader(eval_dataset, lift.tokenizer)  
        evaluator = LIFTEvaluator(lift)  
        results[model_info["name"]] = evaluator.evaluate_dataset(data_loader)  
    
    # 分析结果  
    model_comparison = []  
    for model_name, model_results in results.items():  
        model_comparison.append({  
            "model": model_name,  
            "exact_match": model_results["lift"]["exact_match"],  
            "f1_score": model_results["lift"]["f1_score"],  
            "rouge_l": model_results["lift"]["rouge_l"]  
        })  
    
    # 可视化比较  
    visualize_model_comparison(model_comparison)  
    
    return results  



if __name__ == "__main__":  
    results = evaluate_lift_model()  
    print("评测完成!")  