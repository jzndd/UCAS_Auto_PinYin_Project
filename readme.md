## 项目简介

略

## 思路

### 1. 字库匹配 + piyinyin

detail......

#### 2. LSTM来做，

+ step 1 : 爬虫获得文本，并通过正则表达式进行数据清洗 

+ step 2 ：使用 字库匹配 + pypinyin 自动注音  (method1/method1.py)

    + 存在的问题： 不可解释 ？ 太 easy ？

+ step 3 : 从注音文本中使用正则表达式获得多音字，和其发音，获得模型的训练数据 (method2/get_data.py)

    + step3.1 : 添加进 train_data 中的规则 (train_data[word] 中此发音的样本数量小于等于5) 
        (原因是因为：避免样本过多 / 正反样例失衡)


+ step 4 : 为每一个多音字样本进行创建一个模型，最终使用 nn.ModuleDict 对模型进行打包，合并为一个模型 (method2/train.py)

+ step 5 : 测试模型 / 使用模型进行推理 (method2/test.py)

#### 3. 基于规则模型来做，

文献有现成的，但是都是20世纪初的文献

### git使用

git pull

修改代码后 ，再次 git pull ,有冲突先 git stash 然后再 git pull ，最后git push

## 从文档/介绍方法的角度，可以是这个思路

+ 介绍问题定义，问题背景，问题意义

+ 方法一 ： 简略介绍 pinyin 库

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
        会出现两个问题 ： 1）贵 ； 2）生成质量不稳定，如下图，需要大量的人工精力去修改
        ![](assets/LLM-wrong_example.png)
        ![](assets/LLM_wrong_example2.png)

        考虑使用爬虫 + pypinyin 自动注音的方式获得数据集，最终爬取了 3.7 MB的文本数据，使用 pypinyin 添加了注音后进行**数据集构建**，数据集构建的代码为 ： method2/get_data.py

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
        为什么这样做 ？
    
        最终从 3.7 M 的文本中获得了 509 KB 的数据集

    + 模型训练 (method2/train.py)


        每个网络 （DisambiguationLSTM）是一层 Embedding  + LSTM

        为每个模型训练一个 DisambiguationLSTM ， 最终使用 nn.ModuleDict 的方式做模型集成

        （报告中在这里可以绘制 loss 曲线 ？）

    + 模型推理 (method2/test.py)

        1）正则匹配 （，。；：“ ”）切割句子

        2）检查多音字 list 在不在句子中

        3）在句子中的话，根据此字和切割出来的句子做推理，然后注音

        4）更新句子长度

    + UI
 

