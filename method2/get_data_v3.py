import pandas as pd
import json
import sys

sys.path.append('..')

# 读取 CSV 文件
df = pd.read_csv('test/train.csv')

# 初始化字典
result = {}

# 遍历每一行，提取多音字
for _, row in df.iterrows():
    text = row['text']
    labels = eval(row['label'])  # 将字符串转换为列表
    # 遍历标签并提取多音字
    for i, label in enumerate(labels):
        if label != 'NA':
            # 只提取多音字汉字
            hanzi = text[i]
            if hanzi not in result:
                result[hanzi] = []
            result[hanzi].append([text, hanzi, label])

# 将结果保存为 JSON 文件
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, ensure_ascii=False, indent=4)
