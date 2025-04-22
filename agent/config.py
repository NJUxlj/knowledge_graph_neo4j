
from typing import Dict, List, Literal


class Config:
    context_compression_method:Literal["lift","semantic"]="semantic" # 压缩方法，默认为语义压缩