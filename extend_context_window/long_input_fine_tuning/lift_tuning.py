import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader  
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments  
from typing import List, Dict, Tuple, Optional, Union  
import random  
import numpy as np  
import json  
import os  
from tqdm import tqdm  



'''
代码说明


1. LIFTConfig
配置类，包含所有LIFT模型训练所需的超参数，如基础模型名称、上下文窗口大小、分段长度、重叠长度等。

2. GatedMemory
论文中提出的门控记忆模块，用于自动平衡长输入记忆和ICL能力。它通过一个门控机制来控制原始模型输出和记忆上下文的融合比例。

3. LIFTModel
LIFT的核心模型实现，在基础语言模型上添加门控记忆模块，使短上下文模型能够适应长输入。

4. LongInputDataset
处理长输入文本的数据集类，实现了分段处理长输入和辅助QA任务的构建。

5. QAGenerator
辅助QA对生成器，用于根据长文本生成问答对，作为辅助任务。

6. LIFT
主类，整合了所有组件，提供训练和推理接口。

核心功能特点
分段处理长输入：将长输入分割成可以放入短上下文窗口的重叠段落，减少计算复杂度。

上下文化训练：在相似的语义空间内对齐输入记忆任务和辅助问答任务，统一训练和测试格式。

辅助问答任务：通过生成并训练基于长输入的QA对，增强模型对长输入的理解和推理能力。

门控记忆机制：专门设计的注意力适配器，能自动平衡长输入记忆和原始上下文学习能力。

高效推理：在推理时，可以结合截断上下文和参数中存储的知识，提高长文本理解能力。


'''



class LIFTConfig:  
    """LIFT模型配置"""  
    def __init__(  
        self,  
        base_model_name: str = "meta-llama/Llama-3-8B-Instruct",  
        max_length: int = 4096,  # 模型上下文窗口大小  
        segment_length: int = 2048,  # 分段长度  
        overlap_length: int = 512,  # 段落重叠长度  
        num_qa_pairs: int = 10,  # 辅助QA对数量  
        use_gated_memory: bool = True,  # 是否使用门控记忆  
        use_contextualized_training: bool = True,  # 是否使用上下文化训练  
        gated_memory_dim: int = 256,  # 门控记忆维度  
        learning_rate: float = 1e-5,  
        output_dir: str = "./lift_output",  
        random_seed: int = 42  
    ):  
        self.base_model_name = base_model_name  
        self.max_length = max_length  
        self.segment_length = segment_length  
        self.overlap_length = overlap_length  
        self.num_qa_pairs = num_qa_pairs  
        self.use_gated_memory = use_gated_memory  
        self.use_contextualized_training = use_contextualized_training  
        self.gated_memory_dim = gated_memory_dim  
        self.learning_rate = learning_rate  
        self.output_dir = output_dir  
        self.random_seed = random_seed  


class GatedMemory(nn.Module):  
    """门控记忆模块  
    
    用于自动平衡长输入记忆和原始上下文学习能力  
    """  
    def __init__(self, hidden_size, memory_dim):  
        super().__init__()  
        self.hidden_size = hidden_size  
        self.memory_dim = memory_dim  
        
        # 门控机制  
        self.gate_proj = nn.Linear(hidden_size, 1, bias=False)  
        
        # 记忆投影  
        self.memory_key = nn.Linear(hidden_size, memory_dim, bias=False)  
        self.memory_value = nn.Linear(hidden_size, hidden_size, bias=False)  
        
        # 初始化参数  
        nn.init.normal_(self.gate_proj.weight, std=0.02)  
        nn.init.normal_(self.memory_key.weight, std=0.02)  
        nn.init.normal_(self.memory_value.weight, std=0.02)  
    
    def forward(self, hidden_states):  
        # 计算门控值 (batch_size, seq_len, 1)  
        gate = torch.sigmoid(self.gate_proj(hidden_states))  
        
        # 计算记忆键和值  
        memory_keys = self.memory_key(hidden_states)  # (batch_size, seq_len, memory_dim)  
        memory_values = self.memory_value(hidden_states)  # (batch_size, seq_len, hidden_size)  
        
        # 计算注意力分数  
        attn_scores = torch.matmul(memory_keys, memory_keys.transpose(-1, -2))  # (batch_size, seq_len, seq_len)  
        attn_scores = attn_scores / (self.memory_dim ** 0.5)  # 缩放  
        attn_probs = F.softmax(attn_scores, dim=-1)  
        
        # 获取记忆上下文  
        memory_context = torch.matmul(attn_probs, memory_values)  # (batch_size, seq_len, hidden_size)  
        
        # 应用门控机制  
        output = hidden_states * (1 - gate) + memory_context * gate  
        
        return output  


class LIFTModel(nn.Module):  
    """LIFT模型实现"""  
    def __init__(self, config: LIFTConfig):  
        super().__init__()  
        self.config = config  
        
        # 加载基础模型和分词器  
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)  
        self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name)  
        self.hidden_size = self.model.config.hidden_size  
        
        # 设置为训练模式  
        self.model.train()  
        
        # 门控记忆模块  
        if config.use_gated_memory:  
            # 为每一层Transformer添加门控记忆  
            self.gated_memories = nn.ModuleList([  
                GatedMemory(self.hidden_size, config.gated_memory_dim)   
                for _ in range(len(self.model.model.layers))  
            ])  
            
            # 修改Transformer层的前向传播函数  
            for i, layer in enumerate(self.model.model.layers):  
                orig_forward = layer.forward  
                gated_memory = self.gated_memories[i]  
                
                # 重载前向传播函数，加入门控记忆处理  
                def new_forward(self_layer, hidden_states, *args, **kwargs):  
                    # 原始层的处理  
                    outputs = orig_forward(hidden_states, *args, **kwargs)  
                    # 添加门控记忆  
                    outputs = gated_memory(outputs[0])  
                    return (outputs,) + outputs[1:]  
                
                # 绑定新的前向传播函数  
                layer.forward = new_forward.__get__(layer, type(layer))  
    
    def get_trainable_params(self):  
        """获取可训练的参数"""  
        if self.config.use_gated_memory:  
            # 仅对门控记忆模块的参数进行训练  
            return [p for n, p in self.named_parameters() if "gated_memories" in n]  
        else:  
            # 对整个模型参数进行训练  
            return self.parameters()  
    
    def forward(self, input_ids, attention_mask=None, labels=None):  
        """前向传播"""  
        return self.model(  
            input_ids=input_ids,  
            attention_mask=attention_mask,  
            labels=labels  
        )  


class LongInputDataset(Dataset):  
    """长输入数据集"""  
    def __init__(  
        self,   
        tokenizer,   
        long_text: str,   
        config: LIFTConfig,  
        qa_pairs: List[Dict[str, str]] = None  
    ):  
        self.tokenizer = tokenizer  
        self.config = config  
        
        # 对长文本进行分词  
        self.long_text_tokens = self.tokenizer.encode(long_text)  
        
        # 计算段落数量  
        step_size = config.segment_length - config.overlap_length  
        num_segments = max(1, (len(self.long_text_tokens) - config.overlap_length) // step_size)  
        
        # 生成段落  
        self.segments = []  
        for i in range(num_segments):  
            start_idx = i * step_size  
            end_idx = min(start_idx + config.segment_length, len(self.long_text_tokens))  
            self.segments.append(self.long_text_tokens[start_idx:end_idx])  
        
        # 辅助QA对  
        self.qa_pairs = qa_pairs if qa_pairs else []  
        
        # 构建数据集样本  
        self.examples = []  
        
        # 对每个段落构建语言建模样本  
        for segment in self.segments:  
            # 如果启用上下文化训练，则随机添加上下文  
            if config.use_contextualized_training:  
                # 从长文本开头和结尾随机选择上下文  
                context_tokens = self._sample_context(self.long_text_tokens)  
                
                # 构建完整输入：上下文 + 段落  
                tokens = context_tokens + segment  
                # 确保不超过最大长度  
                if len(tokens) > config.max_length:  
                    tokens = tokens[-config.max_length:]  
            else:  
                tokens = segment  
                # 确保不超过最大长度  
                if len(tokens) > config.max_length:  
                    tokens = tokens[:config.max_length]  
            
            self.examples.append({  
                "input_ids": tokens,  
                "labels": tokens.copy(),  # 语言建模任务的标签与输入相同  
                "task_type": "lm"  # 标记为语言建模任务  
            })  
        
        # 对辅助QA对构建监督微调样本  
        for qa_pair in self.qa_pairs:  
            question = qa_pair["question"]  
            answer = qa_pair["answer"]  
            
            if config.use_contextualized_training:  
                # 从长文本开头和结尾随机选择上下文  
                context_tokens = self._sample_context(self.long_text_tokens)  
                
                # 构建完整输入：上下文 + 问题 + 答案  
                question_tokens = self.tokenizer.encode(question)  
                answer_tokens = self.tokenizer.encode(answer)  
                
                # 输入是上下文+问题，标签是答案  
                input_tokens = context_tokens + question_tokens  
                # 确保不超过最大长度  
                if len(input_tokens) > config.max_length:  
                    input_tokens = input_tokens[-config.max_length:]  
                
                self.examples.append({  
                    "input_ids": input_tokens,  
                    "labels": answer_tokens,  
                    "task_type": "qa"  # 标记为QA任务  
                })  
            else:  
                # 构建完整输入：问题 + 答案  
                prompt = f"{question}\n{answer}"  
                tokens = self.tokenizer.encode(prompt)  
                
                # 计算问题部分的长度  
                question_len = len(self.tokenizer.encode(question))  
                
                # 构建标签：前面的问题部分为-100（忽略），后面的答案部分为实际标签  
                labels = [-100] * question_len + tokens[question_len:]  
                
                # 确保不超过最大长度  
                if len(tokens) > config.max_length:  
                    tokens = tokens[:config.max_length]  
                    labels = labels[:config.max_length]  
                
                self.examples.append({  
                    "input_ids": tokens,  
                    "labels": labels,  
                    "task_type": "qa"  # 标记为QA任务  
                })  
    
    def _sample_context(self, tokens):  
        """从长文本开头和结尾随机采样上下文"""  
        # 计算上下文长度（最大长度的一半）  
        context_len = self.config.max_length // 4  
        
        # 从文档开头采样  
        prefix_len = min(context_len // 2, len(tokens) // 4)  
        prefix = tokens[:prefix_len]  
        
        # 从文档结尾采样  
        suffix_len = min(context_len // 2, len(tokens) // 4)  
        suffix = tokens[-suffix_len:] if suffix_len > 0 else []  
        
        return prefix + suffix  
    
    def __len__(self):  
        return len(self.examples)  
    
    def __getitem__(self, idx):  
        return self.examples[idx]  


def collate_fn(batch):  
    """数据批次整理函数"""  
    input_ids = [item["input_ids"] for item in batch]  
    labels = [item["labels"] for item in batch]  
    
    # 找出最大长度  
    max_length = max([len(ids) for ids in input_ids])  
    
    # 填充输入  
    input_ids_padded = []  
    attention_mask = []  
    labels_padded = []  
    
    for ids, lbl in zip(input_ids, labels):  
        # 填充输入  
        padding_length = max_length - len(ids)  
        input_ids_padded.append(ids + [0] * padding_length)  
        attention_mask.append([1] * len(ids) + [0] * padding_length)  
        
        # 填充标签  
        if isinstance(lbl, list) and len(lbl) == len(ids):  
            # QA任务，部分标签可能为-100  
            labels_padded.append(lbl + [-100] * padding_length)  
        else:  
            # 语言建模任务，标签与输入相同  
            labels_padded.append(ids + [-100] * padding_length)  
    
    return {  
        "input_ids": torch.tensor(input_ids_padded),  
        "attention_mask": torch.tensor(attention_mask),  
        "labels": torch.tensor(labels_padded)  
    }  


class QAGenerator:  
    """辅助QA对生成器"""  
    def __init__(self, model_name="meta-llama/Llama-3-8B-Instruct"):  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.model = AutoModelForCausalLM.from_pretrained(model_name)  
        
    def generate_qa_pairs(self, text: str, num_pairs: int = 10) -> List[Dict[str, str]]:  
        """基于文本生成QA对"""  
        # 以实际应用中，可以使用API服务或更强大的模型来生成QA对  
        # 以下是简化的实现  
        qa_pairs = []  
        
        # 将长文本分割成段落  
        paragraphs = [p for p in text.split('\n\n') if p.strip()]  
        
        # 如果段落太少，直接使用整个文本  
        if len(paragraphs) < num_pairs:  
            paragraphs = [text]  
        
        # 随机选择段落生成QA对  
        selected_paragraphs = random.sample(paragraphs, min(num_pairs, len(paragraphs)))  
        
        for paragraph in selected_paragraphs:  
            # 构建提示  
            prompt = f"""Based on the following text, generate a question and its answer that tests understanding of the content.  
            
Text: {paragraph}  

Question:"""  
            
            # 进行推理  
            inputs = self.tokenizer(prompt, return_tensors="pt")  
            outputs = self.model.generate(  
                inputs["input_ids"],  
                max_new_tokens=100,  
                temperature=0.7,  
                num_return_sequences=1  
            )  
            
            # 解码输出  
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
            
            # 提取问题和答案  
            try:  
                # 移除原始提示  
                generated_text = generated_text.replace(prompt, "")  
                
                # 分离问题和答案  
                if "Answer:" in generated_text:  
                    question, answer = generated_text.split("Answer:", 1)  
                    question = question.strip()  
                    answer = answer.strip()  
                    
                    qa_pairs.append({  
                        "question": question,  
                        "answer": answer  
                    })  
            except:  
                # 处理错误格式  
                continue  
        
        return qa_pairs  


class LIFT:  
    """LIFT主类"""  
    def __init__(self, config: LIFTConfig):  
        self.config = config  
        
        # 设置随机种子  
        random.seed(config.random_seed)  
        np.random.seed(config.random_seed)  
        torch.manual_seed(config.random_seed)  
        
        # 初始化模型  
        self.model = LIFTModel(config)  
        self.tokenizer = self.model.tokenizer  
        
        # 初始化QA生成器  
        self.qa_generator = QAGenerator(config.base_model_name)  
    
    def train(self, long_text: str, num_epochs: int = 3, batch_size: int = 8):  
        """训练LIFT模型"""  
        print("开始LIFT训练过程...")  
        print(f"长文本长度: {len(long_text)} 字符")  
        
        # 生成辅助QA对  
        print(f"生成辅助QA对 ({self.config.num_qa_pairs} 个)...")  
        qa_pairs = self.qa_generator.generate_qa_pairs(  
            long_text,   
            num_pairs=self.config.num_qa_pairs  
        )  
        
        # 创建数据集  
        print("创建训练数据集...")  
        dataset = LongInputDataset(  
            self.tokenizer,  
            long_text,  
            self.config,  
            qa_pairs  
        )  
        
        # 创建数据加载器  
        dataloader = DataLoader(  
            dataset,  
            batch_size=batch_size,  
            shuffle=True,  
            collate_fn=collate_fn  
        )  
        
        # 配置优化器  
        optimizer = torch.optim.AdamW(  
            self.model.get_trainable_params(),  
            lr=self.config.learning_rate  
        )  
        
        # 训练循环  
        print(f"开始训练 ({num_epochs} 个周期)...")  
        self.model.train()  
        for epoch in range(num_epochs):  
            total_loss = 0  
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):  
                # 转移数据到设备  
                batch = {k: v.to(self.model.model.device) for k, v in batch.items()}  
                
                # 前向传播  
                outputs = self.model(**batch)  
                loss = outputs.loss  
                
                # 反向传播  
                loss.backward()  
                optimizer.step()  
                optimizer.zero_grad()  
                
                total_loss += loss.item()  
            
            avg_loss = total_loss / len(dataloader)  
            print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")  
        
        # 保存模型  
        os.makedirs(self.config.output_dir, exist_ok=True)  
        self.model.model.save_pretrained(self.config.output_dir)  
        self.tokenizer.save_pretrained(self.config.output_dir)  
        
        print(f"LIFT训练完成，模型已保存到 {self.config.output_dir}")  
        return self.model  
    
    def answer_question(self, question: str, long_text: str = None, max_new_tokens: int = 100):  
        """使用LIFT模型回答问题"""  
        self.model.eval()  
        
        # 准备输入  
        if long_text:  
            # 如果提供了长文本，则从中截取部分作为上下文  
            # 取开头和结尾，确保总长度不超过上下文窗口限制  
            max_ctx_len = self.config.max_length - 100  # 为问题和生成保留空间  
            
            # 对长文本进行分词  
            long_text_tokens = self.tokenizer.encode(long_text)  
            
            # 从开头和结尾截取  
            prefix_len = max_ctx_len // 2  
            suffix_len = max_ctx_len - prefix_len  
            
            prefix = long_text_tokens[:prefix_len]  
            suffix = long_text_tokens[-suffix_len:] if len(long_text_tokens) > suffix_len else []  
            
            # 构建完整提示：截断上下文 + 问题  
            context_tokens = prefix + suffix  
            prompt = self.tokenizer.decode(context_tokens) + "\n\n" + question  
        else:  
            # 如果没有提供长文本，直接使用问题  
            prompt = question  
        
        # 进行推理  
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.model.device)  
        with torch.no_grad():  
            outputs = self.model.model.generate(  
                inputs["input_ids"],  
                max_new_tokens=max_new_tokens,  
                temperature=0.7,  
                num_return_sequences=1  
            )  
        
        # 解码输出  
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        
        # 移除原始提示，只返回生成的答案  
        answer = generated_text[len(prompt):].strip()  
        
        return answer  


def main():  
    """主函数示例"""  
    # 示例长文本  
    long_text = """  
    [这里是一个很长的文本，比如一篇长文章或文档。实际应用中这部分可能有数万到数十万词。]  
    """  
    
    # 创建LIFT配置  
    config = LIFTConfig(  
        base_model_name="meta-llama/Llama-3-8B-Instruct",  # 使用适合的模型名称  
        max_length=4096,  
        segment_length=2048,  
        overlap_length=512,  
        num_qa_pairs=10,  
        use_gated_memory=True,  
        use_contextualized_training=True,  
        output_dir="./lift_model"  
    )  
    
    # 初始化LIFT  
    lift = LIFT(config)  
    
    # 训练LIFT模型  
    lift.train(long_text, num_epochs=3, batch_size=4)  
    
    # 使用LIFT模型回答问题  
    question = "请总结一下文章的主要内容？"  
    answer = lift.answer_question(question, long_text)  
    print(f"问题: {question}")  
    print(f"答案: {answer}")  


if __name__ == "__main__":  
    main()  