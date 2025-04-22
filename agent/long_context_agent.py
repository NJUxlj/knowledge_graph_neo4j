import logging 
from typing import Dict, List, Any, Tuple, Optional, Literal
from .config import Config
import os, json

from extend_context_window.semantic_compression.loader import DataManager  
from extend_context_window.semantic_compression.model import SemanticCompressor  
from extend_context_window.semantic_compression.evaluate import CompressionEvaluator  
from extend_context_window.semantic_compression.predict import LLMPredictor



class LongContextAgent:  
    """  
    长上下文处理代理类，根据指定方法选择合适的处理策略  
    """  
    
    SUPPORTED_METHODS = {  
        "semantic": "语义压缩方法",  
        "lift": "长输入微调方法"  
    }  
    
    def __init__(self, config=None):  
        """  
        初始化长上下文代理  
        
        Args:  
            config: 配置对象，如果为None则使用默认配置  
            context_compression_method: 上下文压缩方法，可选值见SUPPORTED_METHODS  
        """  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
        
        # 使用提供的配置或创建默认配置  
        self.config = config if config else Config()  
        
        # 验证并设置压缩方法  
        if self.config.context_compression_method not in self.SUPPORTED_METHODS:  
            supported = ", ".join(self.SUPPORTED_METHODS.keys())  
            raise ValueError(f"不支持的压缩方法: {self.config.context_compression_method}。支持的方法: {supported}")  
        
        self.compression_method = self.config.context_compression_method  
        self.logger.info(f"使用{self.SUPPORTED_METHODS[self.context_compression_method]}处理长上下文")  
        
        # 初始化数据管理器  
        self.data_manager = DataManager(self.config)  
        
        # 根据选择的方法初始化相应的组件  
        if self.context_compression_method == "semantic":  
            self.processor = SemanticCompressor(self.config)  
            self.evaluator = CompressionEvaluator(self.config)  
        elif self.context_compression_method == "lift":  
            self.processor = LongInputFineTuner(self.config)  
        
        # 预测器是通用的，不管使用哪种压缩方法  
        self.predictor = LLMPredictor(self.config)  
    
    def process_long_context(self,   
                             document_path: str,   
                             output_dir: str = "output",  
                             evaluate: bool = False,  
                             run_prediction: bool = False,  
                             questions: List[str] = None) -> Dict[str, Any]:  
        """  
        处理长上下文文档  
        
        Args:  
            document_path: 文档路径  
            output_dir: 输出目录  
            evaluate: 是否评估  
            run_prediction: 是否运行预测  
            questions: 问题列表（如果run_prediction为True）  
            
        Returns:  
            处理结果字典  
        """  
        self.logger.info(f"开始处理文档: {document_path}")  
        
        # 加载并预处理文档  
        document_data = self.data_manager.process_document(document_path)  
        
        # 根据选择的方法处理文档  
        if self.compression_method == "semantic_compression":  
            # 使用语义压缩方法  
            processed_result = self.processor.compress_content(document_data['initial_chunks'])  
            
            # 创建输出目录  
            if not os.path.exists(output_dir):  
                os.makedirs(output_dir)  
            
            # 保存压缩结果  
            self._save_semantic_compression_result(processed_result, output_dir)  
            
            # 评估  
            if evaluate:  
                evaluation_results = self.evaluator.evaluate_compression(  
                    document_data['raw_text'],  
                    processed_result  
                )  
                self.evaluator.save_evaluation_results(evaluation_results, output_dir)  
            else:  
                evaluation_results = None  
            
            # 预测  
            if run_prediction:  
                prediction_results = self._run_prediction(  
                    processed_result['compressed_content'],   
                    questions,   
                    output_dir  
                )  
            else:  
                prediction_results = None  
                
            return {  
                "method": "semantic_compression",  
                "processed_result": processed_result,  
                "evaluation_results": evaluation_results,  
                "prediction_results": prediction_results,  
                "document_data": document_data  
            }  
            
        elif self.compression_method == "long_input_fine_tuning":  
            # 使用长输入微调方法  
            processed_result = self.processor.process_document(document_data)  
            
            # 评估  
            if evaluate:  
                evaluation_results = self.processor.evaluate(document_data, processed_result)  
                self.processor.save_results(processed_result, evaluation_results, output_dir)  
            else:  
                evaluation_results = None  
                self.processor.save_results(processed_result, None, output_dir)  
            
            # 预测  
            if run_prediction:  
                prediction_results = self._run_prediction(  
                    processed_result['processed_content'],   
                    questions,   
                    output_dir  
                )  
            else:  
                prediction_results = None  
                
            return {  
                "method": "long_input_fine_tuning",  
                "processed_result": processed_result,  
                "evaluation_results": evaluation_results,  
                "prediction_results": prediction_results,  
                "document_data": document_data  
            }  
    
    def _save_semantic_compression_result(self, result: Dict[str, Any], output_dir: str):  
        """保存语义压缩结果"""  
        # 保存压缩内容  
        with open(os.path.join(output_dir, 'compressed_content.txt'), 'w') as f:  
            f.write(result['compressed_content'])  
        
        # 保存主题映射  
        with open(os.path.join(output_dir, 'topic_mapping.json'), 'w') as f:  
            json.dump(result['topic_map'], f, indent=4)  
        
        # 保存压缩统计信息  
        stats = {  
            'method': 'semantic_compression',  
            'compression_ratio': result['compression_ratio'],  
            'original_chunks_count': len(result['original_chunks']),  
            'compressed_topics_count': len(result['compressed_topics'])  
        }  
        with open(os.path.join(output_dir, 'compression_stats.json'), 'w') as f:  
            json.dump(stats, f, indent=4)  
    
    def _run_prediction(self,   
                       processed_content: str,   
                       questions: List[str],   
                       output_dir: str) -> Dict[str, Any]:  
        """运行预测任务"""  
        self.logger.info("运行预测任务")  
        
        results = {}  
        
        # 提取关键信息  
        key_points = self.predictor.extract_key_information(processed_content)  
        
        # 保存关键信息  
        with open(os.path.join(output_dir, 'key_points.json'), 'w') as f:  
            json.dump(key_points, f, indent=4)  
        
        results["key_points"] = key_points  
        
        # 问答任务  
        if questions:  
            qa_results = self.predictor.run_qa_task(processed_content, questions)  
            self.predictor.save_prediction_results(qa_results, output_dir)  
            results["qa_results"] = qa_results  
        
        return results  