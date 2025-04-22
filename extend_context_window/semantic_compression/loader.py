"""  
负责文档加载和预处理  
"""  

import os  
import re  
import nltk  
import logging  
import PyPDF2  
import docx  
from typing import List, Dict, Any, Union, Tuple  


from .config import Config

# 下载必要的NLTK数据  
try:  
    nltk.data.find('tokenizers/punkt')  
except LookupError:  
    nltk.download('punkt')  

class DocumentLoader:  
    """文档加载和预处理类"""  
    
    def __init__(self):  
        self.supported_formats = {  
            '.txt': self._load_txt,  
            '.pdf': self._load_pdf,  
            '.docx': self._load_docx,  
            '.md': self._load_txt,  
        }  
        logging.basicConfig(level=logging.INFO)  
        self.logger = logging.getLogger(__name__)  
    
    def load_document(self, file_path: str) -> str:  
        """  
        加载文档并返回文本内容  
        
        Args:  
            file_path: 文档路径  
            
        Returns:  
            文档内容字符串  
        """  
        if not os.path.exists(file_path):  
            raise FileNotFoundError(f"文件不存在: {file_path}")  
            
        extension = os.path.splitext(file_path)[1].lower()  
        if extension not in self.supported_formats:  
            raise ValueError(f"不支持的文件格式: {extension}")  
        
        self.logger.info(f"加载文件: {file_path}")  
        content = self.supported_formats[extension](file_path)  
        self.logger.info(f"加载完成, 文档长度: {len(content)} 字符")  
        return content  
    
    def preprocess_text(self, text: str) -> List[str]:  
        """  
        预处理文本，分割成句子  
        
        Args:  
            text: 待处理的文本  
            
        Returns:  
            句子列表  
        """  
        # 清理文本  
        text = re.sub(r'\s+', ' ', text)  # 规范化空白字符  
        text = re.sub(r'\n+', '\n', text)  # 规范化换行  
        
        # 分割句子  
        sentences = nltk.sent_tokenize(text)  
        
        # 过滤空句子  
        sentences = [s.strip() for s in sentences if s.strip()]  
        
        self.logger.info(f"预处理完成, 共 {len(sentences)} 个句子")  
        return sentences  
    
    def chunk_sentences(self, sentences: List[str], chunk_size: int) -> List[str]:  
        """  
        将句子分组成固定大小的块  
        
        Args:  
            sentences: 句子列表  
            chunk_size: 每个块的句子数量  
            
        Returns:  
            文本块列表  
        """  
        chunks = []  
        for i in range(0, len(sentences), chunk_size):  
            chunk = ' '.join(sentences[i:i + chunk_size])  
            chunks.append(chunk)  
        
        self.logger.info(f"文本分块完成, 共 {len(chunks)} 个块")  
        return chunks  
    
    def _load_txt(self, file_path: str) -> str:  
        """加载TXT文件"""  
        with open(file_path, 'r', encoding='utf-8') as f:  
            return f.read()  
    
    def _load_pdf(self, file_path: str) -> str:  
        """加载PDF文件"""  
        text = ""  
        with open(file_path, 'rb') as f:  
            pdf_reader = PyPDF2.PdfReader(f)  
            for page_num in range(len(pdf_reader.pages)):  
                page = pdf_reader.pages[page_num]  
                text += page.extract_text() + "\n"  
        return text  
    
    def _load_docx(self, file_path: str) -> str:  
        """加载DOCX文件"""  
        doc = docx.Document(file_path)  
        text = ""  
        for para in doc.paragraphs:  
            text += para.text + "\n"  
        return text  


class DataManager:  
    """数据管理类，整合加载和预处理步骤"""  
    
    def __init__(self, config:Config):  
        self.config = config  
        self.loader = DocumentLoader()  
    
    def process_document(self, file_path: str) -> Dict[str, Any]:  
        """  
        处理文档，返回准备好的数据  
        
        Args:  
            file_path: 文档路径  
            
        Returns:  
            包含原始文本、句子和初始块的字典  
        """  
        # 加载文档  
        raw_text = self.loader.load_document(file_path)  
        
        # 提取句子  
        sentences = self.loader.preprocess_text(raw_text)  
        
        # 分块  
        initial_chunks = self.loader.chunk_sentences(  
            sentences,   
            self.config.SENTENCE_CHUNK_SIZE  
        )  
        
        return {  
            "raw_text": raw_text,  
            "sentences": sentences,  
            "initial_chunks": initial_chunks,  
            "file_path": file_path  
        }  
        
        
        




if __name__ == '__main__':
    '''
    python -m extend_context_window.semantic_compression.loader
    '''
    config = Config()
    manager = DataManager(config)
    
    path = "extend_context_window\semantic_compression\Extending Context Window of Large Language Models via Semantic Compression.pdf"
    result = manager.process_document(path)
    
    
    print(result)