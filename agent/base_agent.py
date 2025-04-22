import logging

from abc import ABC, abstractmethod

class BaseAgent(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化基础代理...")
        
        
    @abstractmethod
    def run(self, input_data):
        """
        运行代理的主要逻辑
        """