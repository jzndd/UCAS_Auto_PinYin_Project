## 项目简介

略

## git使用

git pull

修改代码后 ，再次 git pull ,

有冲突先 git stash 然后再 git pull , git stash pop ，

最后git push

## 其他文档地址

文档地址 ： https://www.overleaf.com/4464365427cnnrmvnwtssc#66a4ab

pth 地址 ：

model_v1 :  百度网盘分享的文件disambiguation_models_big.pth 链接：https://pan.baidu.com/s/1LeAreewqKOikbh5nleoejA?pwd=4qex  提取码：4qex ; 谷歌链接：https://drive.google.com/file/d/1Oeg4F-PKiLJf8De4NKUJUWgzI8Xi8NrO/view?usp=drive_link

model_v2: 百度网盘分享disambiguation_models.pth链接：https://pan.baidu.com/s/1UenI6JgfvLRxVjw6qtQG8w?pwd=xrqv 提取码：xrqv ；谷歌链接：https://drive.google.com/file/d/1Oeg4F-PKiLJf8De4NKUJUWgzI8Xi8NrO/view?usp=drive_link

model_v3: 谷歌链接: https://drive.google.com/file/d/1HH53QUr3MphqeJI623mzcRqgTaQsJuY7/view?usp=drive_link

## 代码结构

data 存放**模型文件**（.pth, 需要去我的云盘下载）、**训练数据** （.json）、**测试文档与输出文档** （.docx）以及 pypinyin 的支持库 （polyphone.json）
method1 
|_ method1.py 运行，直接读取docx 文档并注音

method2
|_ get_data_v_.py 获取各种类型数据集的文件，活比较脏，没必要看，处理思路会写在文档里
|_ train.py 训练模型，获得 pth 文件
|_ 加载模型，推理，输入 docx 文件，输出 docx 推理后的文件

## TODO:

LLM 方式作为 method3 合并到代码中
-[] LLM Teacher-Student  @转运使
-[] LLM 语义理解          @郝锐

method2:
-[] 模型文件 v4 的上传 @jzn
-[] 模型推理时，一个句子中有重复的多音字会出现 bug @jzn   （未完成，想摆烂）

UI 合并到仓库中

## 从文档/介绍方法的角度，可以是这个思路

+ 介绍问题定义，问题背景，问题意义

+ 方法一 ： 简略介绍 pypinyin 库

+ 方法二 ： LSTM 

    + 数据来源 ？ 

        构建了如下的数据形式

        ```python
                "了": [
            ('他已经回来了', '了', 'le'),
            ('你去了哪里？', '了', 'le'),
            ('事情终于结束了', '了', 'le'),
            ('我吃完饭了', '了', 'le'),
            ('他笑了起来', '了', 'le'),
            ('问题解决了', '了', 'le'),
            ('我早就知道了', '了', 'le'),
            ('那件事了了', '了', 'liǎo'),
            ('他什么都不明了', '了', 'liǎo'),
            ('事情很快了结', '了', 'liǎo'),
            ],
        ```

        + 大模型生成样本
        给的 prompt 是：
        ```
        '六': ('liù', 'lù'), '切': ('qiē', 'qiè'), '分': ('fēn', 'fèn'),...
        请为这些多音字，每个字生成 10 个句子，样式为：
        "少": [
            ("少年的时光很宝贵", "少", "shào"),
            ("少数服从多数是原则", "少", "shǎo"),
            ("少林寺的武功很厉害", "少", "shào"),
            ("他少言寡语", "少", "shǎo"),
            ("少年强则国强", "少", "shào"),
            ("少部分人持有异议", "少", "shǎo"),
            ("他少不更事", "少", "shào"),
            ("少安毋躁，慢慢来", "少", "shǎo"),
            ("少年的志向很远大", "少", "shào"),
            ("少一点抱怨，多一点行动", "少", "shǎo")
            ],
        ```
        会出现两个问题 ：  1）生成质量不稳定，如下图，需要大量的人工精力去修改
        ![](assets/LLM-wrong_example.png)
        ![](assets/LLM_wrong_example2.png)

        考虑使用爬虫 + pypinyin 自动注音的方式获得数据集，爬取了 3.7 MB的文本数据，使用 pypinyin 添加了注音后进行**数据集构建**，数据集构建的代码为 ： method2/get_data.py

        核心点如下：

        1） 正则表达式实现拼音提取与文字提取 

        2）
        ```python
        # 判断即将添加的发音是否已经存在于 train_data 中且个数大于 5
            if word in train_data:
                existing_pronunciations = [entry[2] for entry in train_data[word]]
                if existing_pronunciations.count(pronunciation) > 5:
                    continue  # 如果发音个数大于5，则跳过添加
        ```
        为什么这样做 ？-- 可以避免样本失衡带来的问题;
    
        最终，分别使用了四类数据集来构建数据：

        1）102KB 大模型 + 人工挑选

        2）500KB pinyin + 爬虫高质量的文本

        3）2.55MB （群聊中的 csv 数据集经过处理）

        4）1.59 MB （群聊中的 sent 文件经过处理）

    + 模型训练 (method2/train.py)
    
        每个网络 （DisambiguationLSTM）是一层 Embedding  + LSTM
        
        为每个字训练一个 DisambiguationLSTM ， 最终使用 nn.ModuleDict 的方式做模型集成
        
        （报告中在这里可以绘制 loss 曲线 ？）
        
    + 模型推理 (method2/test.py)
    
        1）正则匹配 （，。；：“ ”）切割句子
    
        2）检查多音字 list 在不在句子中
    
        3）在句子中的话，根据此字和切割出来的句子做推理，然后注音
    
        4）更新句子长度
    
    + UI

