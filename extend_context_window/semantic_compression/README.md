# 使用说明


### 安装所需依赖：
```bash
pip install torch transformers sentence-transformers nltk PyPDF2 python-docx networkx scikit-learn rouge-score bert-score python-louvain  
```

### 运行主程序：

```bash
# 基本压缩  
python main.py --input your_document.pdf --output_dir ./output  

# 压缩并评估  
python main.py --input your_document.pdf --output_dir ./output --evaluate  

# 压缩并提取关键信息  
python main.py --input your_document.pdf --output_dir ./output --predict  

# 压缩并进行问答  
python main.py \
--input your_document.pdf \
--output_dir ./output \
--predict --questions questions.json  
```

其中，questions.json 是一个包含问题列表的JSON文件，例如：

```json
[  
  "这篇论文的主要贡献是什么？",  
  "论文中提出的方法如何解决长上下文问题？",  
  "该方法的性能如何？"  
]  
```

### 系统特点
- 模块化设计：代码被组织成独立的模块，便于维护和扩展。
- 完整实现：包含论文中描述的所有核心步骤：基于主题的分块、语义压缩和重组。
- 灵活配置：通过配置文件可以调整各种参数，适应不同需求。
- 全面评估：提供多种评估指标，测量压缩效果。
- 预测功能：可以使用压缩后的内容进行问答和信息提取。