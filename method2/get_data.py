import re
from docx import Document

# 初始化空字典
train_data = {}

# 读取docx文件
doc = Document('data/train_pinyin_file.docx')  # 替换为你的文件名

# 遍历文档中的段落
for para in doc.paragraphs:
    text = para.text
    
    # 使用标点符号分隔句子
    sentences = re.split(r'[。；，]', text)
    
    for sentence in sentences:   
        matches = re.findall(r'(\w+)\((\w+)\)', sentence)
        for word, pronunciation in matches:
    
            cleaned_sentence = re.sub(r'\(.*?\)', '', sentence).strip()
            
            # 判断即将添加的发音是否已经存在于 train_data 中且个数大于 5
            if word in train_data:
                existing_pronunciations = [entry[2] for entry in train_data[word]]
                if existing_pronunciations.count(pronunciation) > 5:
                    continue  # 如果发音个数大于5，则跳过添加
            
            if len(word) > 1:
                continue

            # 添加到 train_data
            if word not in train_data:
                train_data[word] = []
            train_data[word].append((cleaned_sentence, word, pronunciation))


# 存储train_data
import json

with open('data/train_data_big.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

print("数据集已存储")