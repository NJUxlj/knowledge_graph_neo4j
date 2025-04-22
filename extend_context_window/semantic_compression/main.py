"""  
主程序入口，整合所有模块  
"""  

import argparse  
import os  
import logging  
import json  
from typing import Dict, Any, Optional  

from .config import Config  
from .loader import DataManager  
from .model import SemanticCompressor  
from .evaluate import CompressionEvaluator  
from .predict import LLMPredictor  

def setup_logging():  
    """设置日志记录"""  
    logging.basicConfig(  
        level=logging.INFO,  
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  
        handlers=[  
            logging.FileHandler("semantic_compression.log"),  
            logging.StreamHandler()  
        ]  
    )  
    return logging.getLogger(__name__)  

def parse_arguments():  
    """解析命令行参数"""  
    parser = argparse.ArgumentParser(description='语义压缩系统')  
    
    parser.add_argument('--input', type=str, required=True, 
                        help='输入文档路径')  
    parser.add_argument('--output_dir', type=str, default='output',  
                        help='输出目录')  
    parser.add_argument('--evaluate', action='store_true',  
                        help='是否评估压缩效果')  
    parser.add_argument('--predict', action='store_true',  
                        help='是否运行预测任务')  
    parser.add_argument('--questions', type=str, default=None,  
                        help='问题文件路径（JSON格式）')  
    
    return parser.parse_args()  

def save_compressed_result(result: Dict[str, Any], output_dir: str):  
    """保存压缩结果"""  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
    
    # 保存压缩内容  
    with open(os.path.join(output_dir, 'compressed_content.txt'), 'w') as f:  
        f.write(result['compressed_content'])  
    
    # 保存主题映射  
    with open(os.path.join(output_dir, 'topic_mapping.json'), 'w') as f:  
        json.dump(result['topic_map'], f, indent=4)  
    
    # 保存压缩统计信息  
    stats = {  
        'compression_ratio': result['compression_ratio'],  
        'original_chunks_count': len(result['original_chunks']),  
        'compressed_topics_count': len(result['compressed_topics'])  
    }  
    with open(os.path.join(output_dir, 'compression_stats.json'), 'w') as f:  
        json.dump(stats, f, indent=4)  

def main():  
    """主函数"""  
    args = parse_arguments()  
    logger = setup_logging()  
    
    logger.info("初始化配置")  
    config = Config()  
    
    # 数据加载和预处理  
    logger.info(f"加载文档: {args.input}")  
    data_manager = DataManager(config)  
    document_data = data_manager.process_document(args.input)  
    
    # 语义压缩  
    logger.info("开始语义压缩")  
    compressor = SemanticCompressor(config)  
    compression_result = compressor.compress_content(document_data['initial_chunks'])  
    
    # 保存压缩结果  
    logger.info(f"保存压缩结果到: {args.output_dir}")  
    save_compressed_result(compression_result, args.output_dir)  
    
    # 评估压缩效果  
    if args.evaluate:  
        logger.info("评估压缩效果")  
        evaluator = CompressionEvaluator(config)  
        evaluation_results = evaluator.evaluate_compression(  
            document_data['raw_text'],  
            compression_result  
        )  
        evaluator.save_evaluation_results(evaluation_results, args.output_dir)  
    
    # 使用压缩内容进行预测  
    if args.predict:  
        logger.info("使用压缩内容进行预测")  
        predictor = LLMPredictor(config)  
        
        # 提取关键信息  
        key_points = predictor.extract_key_information(compression_result['compressed_content'])  
        
        # 保存关键信息  
        with open(os.path.join(args.output_dir, 'key_points.json'), 'w') as f:  
            json.dump(key_points, f, indent=4)  
        
        # 如果提供了问题文件，进行问答任务  
        if args.questions:  
            with open(args.questions, 'r') as f:  
                questions = json.load(f)  
            
            qa_results = predictor.run_qa_task(  
                compression_result['compressed_content'],  
                questions  
            )  
            
            predictor.save_prediction_results(qa_results, args.output_dir)  
    
    logger.info("语义压缩任务完成")  

if __name__ == "__main__":  
    main()  