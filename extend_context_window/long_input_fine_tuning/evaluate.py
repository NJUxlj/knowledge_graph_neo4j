class LIFTEvaluator:  
    """LIFT评测器"""  
    def __init__(self, lift_model, base_model=None):  
        """  
        Args:  
            lift_model: LIFT微调后的模型  
            base_model: 原始基础模型(用于ICL基线比较)  
        """  
        self.lift_model = lift_model  
        self.base_model = base_model if base_model else None  
        self.metrics = {  
            "exact_match": 0,  
            "f1_score": 0,  
            "rouge_l": 0,  
            "accuracy": 0,  
            "total_samples": 0  
        }  
        
    def evaluate_dataset(self, dataset_loader, verbose=True):  
        """评估数据集"""  
        results = {  
            "lift": self.metrics.copy(),  
            "icl": self.metrics.copy() if self.base_model else None,  
            "truncated_icl": self.metrics.copy() if self.base_model else None  
        }  
        
        for sample in tqdm(dataset_loader, desc="Evaluating"):  
            # 1. 评估LIFT模型 (只使用问题)  
            lift_answer = self.lift_model.answer_question(sample["question"])  
            self._update_metrics(results["lift"], lift_answer, sample["ground_truth"])  
            
            # 2. 评估基础模型 (使用完整ICL)  
            if self.base_model:  
                icl_answer = self.base_model.generate_answer(sample["icl_input"])  
                self._update_metrics(results["icl"], icl_answer, sample["ground_truth"])  
                
                # 3. 评估基础模型 (使用截断ICL)  
                truncated_icl_answer = self.base_model.generate_answer(sample["truncated_icl_input"])  
                self._update_metrics(results["truncated_icl"], truncated_icl_answer, sample["ground_truth"])  
            
            # 打印样本级结果  
            if verbose:  
                print(f"Question: {sample['question'][:100]}...")  
                print(f"Ground Truth: {sample['ground_truth'][:100]}...")  
                print(f"LIFT Answer: {lift_answer[:100]}...")  
                if self.base_model:  
                    print(f"ICL Answer: {icl_answer[:100]}...")  
                    print(f"Truncated ICL Answer: {truncated_icl_answer[:100]}...")  
                print("-" * 50)  
        
        # 计算最终指标均值  
        for model_type in results:  
            if results[model_type]:  
                for metric in results[model_type]:  
                    if metric != "total_samples" and results[model_type]["total_samples"] > 0:  
                        results[model_type][metric] /= results[model_type]["total_samples"]  
        
        return results  
    
    def _update_metrics(self, metrics_dict, prediction, ground_truth):  
        """更新评测指标"""  
        metrics_dict["exact_match"] += self._compute_exact_match(prediction, ground_truth)  
        metrics_dict["f1_score"] += self._compute_f1(prediction, ground_truth)  
        metrics_dict["rouge_l"] += self._compute_rouge_l(prediction, ground_truth)  
        metrics_dict["accuracy"] += self._compute_accuracy(prediction, ground_truth)  
        metrics_dict["total_samples"] += 1  
    
    # 实现各种评估指标的计算方法  
    def _compute_exact_match(self, prediction, ground_truth):  
        """计算精确匹配分数"""  
        return 1.0 if self._normalize_text(prediction) == self._normalize_text(ground_truth) else 0.0  
    
    def _compute_f1(self, prediction, ground_truth):  
        """计算F1分数"""  
        # 实现基于词重叠的F1计算  
        pred_tokens = self._normalize_text(prediction).split()  
        truth_tokens = self._normalize_text(ground_truth).split()  
        
        # 计算共同词汇  
        common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)  
        num_common = sum(common.values())  
        
        # 边界情况处理  
        if num_common == 0:  
            return 0.0  
            
        precision = num_common / len(pred_tokens)  
        recall = num_common / len(truth_tokens)  
        f1 = (2 * precision * recall) / (precision + recall)  
        return f1  
    
    def _compute_rouge_l(self, prediction, ground_truth):  
        """计算ROUGE-L分数"""  
        # 利用rouge库计算ROUGE-L  
        try:  
            from rouge import Rouge  
            rouge = Rouge()  
            scores = rouge.get_scores(prediction, ground_truth)  
            return scores[0]["rouge-l"]["f"]  
        except:  
            # 如果rouge库不可用，返回0  
            return 0.0  
    
    def _compute_accuracy(self, prediction, ground_truth):  
        """计算准确率 (适用于分类任务)"""  
        return 1.0 if self._normalize_text(prediction).strip() == self._normalize_text(ground_truth).strip() else 0.0  
    
    def _normalize_text(self, text):  
        """文本规范化处理"""  
        return " ".join(text.lower().split())  