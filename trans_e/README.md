## 如何运行
要使用这个TransE实现，你需要准备好数据文件，然后按照以下步骤操作：

### 准备数据：

创建一个data目录
添加train.txt、valid.txt和test.txt文件
每个文件中，每行格式为：头实体 关系 尾实体

### 训练模型：

```bash
python main.py --data_path ./data/ --embedding_dim 50 --margin 1.0 --learning_rate 0.01 --batch_size 128 --epochs 1000 --distance L1 
``` 

### 预测：
```bash
# 预测尾实体  
python predict.py --model_path ./models/ --head "USA" --relation "has_capital" --mode tail  

# 预测头实体  
python predict.py --model_path ./models/ --relation "has_capital" --tail "Washington" --mode head  

# 预测关系  
python predict.py --model_path ./models/ --head "USA" --tail "Washington" --mode relation  
```