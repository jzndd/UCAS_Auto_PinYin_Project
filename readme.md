# NLP 自动注音项目

该项目是NLP第四小组的自动注音项目，我们开发了一套系统，实现对文档中的多音字和疑难字的自动注音。具体来说，我们的贡献包括：

+ 构建了一个针对docx文档的自动注音训练、测试和评估的完整pipeline。

+ 采用了pypinyin库、训练LSTM神经网络和使用LLM agent三种方法，实现文本的自动注音。

+ 创建了四种数据集，探讨数据集质量对LSTM网络在解决自动注音问题中的关键作用。

+ 通过实验发现，无论是深度学习还是LLM agent，分词和语义理解在自动注音的成功率中发挥了重要作用。

+ 整理了一个汉字单字字频数据集，有效利用规则信息为文本中的易读错字提供注音。

+ 构建了常规样例和困难样例，发现 LSTM 网络和 LLM agent 都取得了优秀的表现，其中，LLM 在困难样例下表现最佳。

+ 设计了一个基于Flask的用户界面，通过清晰的界面和直观的交互，用户可以轻松上传文件和输入文本，获取处理结果。

## 代码结构

data 存放**模型文件**（.pth, 需要去我的云盘下载）、**训练数据** （.json）、**测试文档与输出文档** （.docx）以及 pypinyin 的支持库 （polyphone.json）  

method1   
|_ method1.py 运行，直接读取docx 文档并注音
|_ eval.py 运行，获取方法1的测试正确率

method2  
|_ get_data_v_.py 获取各种类型数据集的文件，活比较脏，没必要看，处理思路会写在文档里  
|_ train.py 训练模型，获得 pth 文件  
|_ test.py 加载模型，推理，输入 docx 文件，输出 docx 推理后的文件
|_ nlp_nan.py 处理难字
|_ eval.py 运行，获取方法2的测试正确率

method3
|_ method3.py 使用大模型推理获得文本的注音

test/, assets/  该文件夹被添加倒 .gitignore 中，属于中间过程的处理文件，无实际意义

## 权重文件

pth 地址 (数据集来源不一致) ：
model_v1 :  百度网盘分享的文件disambiguation_models_big.pth 链接：https://pan.baidu.com/s/1LeAreewqKOikbh5nleoejA?pwd=4qex  提取码：4qex ; 谷歌链接：https://drive.google.com/file/d/1Oeg4F-PKiLJf8De4NKUJUWgzI8Xi8NrO/view?usp=sharing    

model_v2: 百度网盘分享disambiguation_models.pth链接：https://pan.baidu.com/s/1UenI6JgfvLRxVjw6qtQG8w?pwd=xrqv 提取码：xrqv ；谷歌链接：https://drive.google.com/file/d/10Q4K0QeZ8C2XAARqZCxU0MdSzKcYMZnc/view?usp=sharing  

model_v3:https://drive.google.com/file/d/1sXgFdIoOhWKf9gKM4rULt1MiQUdIt9pP/view?usp=sharing  

model_v4: https://drive.google.com/file/d/1UcPRuwHs1hkpPFUJHwctbMKGspLHHukk/view?usp=sharing