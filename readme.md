## 项目简介

略

## 思路

### 1. 字库匹配 + piyinyin

detail......

#### 2. LSTM来做，

+ step 1 : 爬虫获得文本，并通过正则表达式进行数据清洗 

+ step 2 ：使用 字库匹配 + pypinyin 自动注音  (method1/method1.py)

+ step 3 : 从注音文本中使用正则表达式获得多音字，和其发音，获得模型的训练数据 (method2/get_data.py)

    + step3.1 : 添加进 train_data 中的规则 (train_data[word] 中此发音的样本数量小于等于5) 
        (原因是因为：避免样本过多 / 正反样例失衡)

+ step 4 : 为每一个多音字样本进行创建一个模型，最终使用 nn.ModuleDict 对模型进行打包，合并为一个模型 (method2/train.py)

+ step 5 : 测试模型 / 使用模型进行推理 (method2/test_model.py)

#### 3. 基于规则模型来做，

文献有现成的，但是都是20世纪初的文献

### git使用

git pull

修改代码后 ，再次 git pull ,有冲突先 git stash 然后再 git pull ，最后git push
