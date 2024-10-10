import json
import re
import sys

sys.path.append("..")
# 读取文字文件
with open('test/train_v4.sent', 'r', encoding='utf-8') as f:
    text_lines = f.readlines()

# 读取注音文件
with open('test/train_v4.lb', 'r', encoding='utf-8') as f:
    pinyin_lines = f.readlines()

# 初始化字典
result = {}

# 遍历每一行文字
for text_line, pinyin_line in zip(text_lines, pinyin_lines):
    # 移除下划线，获取原句
    original_text = text_line.replace('▁', '').strip()
    # 获取注音列表
    pinyin_list = pinyin_line.strip().split()
    
    # 分割文字行，提取带注音的单个字符
    words = text_line.split('▁')
    
    # 遍历每个字符和对应的注音
    for i, word in enumerate(words):
        if len(word) == 1:  # 只提取长度为1的单个字符
            char = word
            pinyin = pinyin_list[i] if i < len(pinyin_list) else None
            
            if pinyin and char not in result:
                result[char] = []
            if pinyin:
                # 删除所添加字符的左右两个下划线
                entry_text = original_text.replace('▁', '')  # 去除所有下划线
                result[char].append([entry_text, char, pinyin])

# 将结果保存为 JSON 文件
with open('output.json', 'w', encoding='utf-8') as json_file:
    json.dump(result, json_file, ensure_ascii=False, indent=4)


# 后续一些很脏的处理代码，没必要看
# import json
# import sys
# sys.path.append('..')

# # 读取文件
# data = json.load(open('data/train_data_v4_oral.json', 'r', encoding='utf-8'))

# # 读取标准发音数据
# standard_pronunciation = json.load(open('data/polyphone_converted_data.json', 'r', encoding='utf-8'))

# # 将标准发音数据转换为字典，便于检索
# standard_dict = {entry['char']: entry['pinyin'] for entry in standard_pronunciation}

# # 处理数据
# new_data = {}
# for key, values in data.items():
#     pronunciation_count = {}
#     limited_values = []
    
#     # 统计每个发音的出现次数
#     for item in values:
#         pronunciation = item[-1]
#         if pronunciation not in pronunciation_count:
#             pronunciation_count[pronunciation] = 0
#         pronunciation_count[pronunciation] += 1
        
#     # 只保留出现超过一次的发音对应的条目，并检查发音是否在标准发音中
#     for item in values:
#         pronunciation = item[-1]
        
#         # 检查发音是否在标准发音列表中
#         if pronunciation_count[pronunciation] > 8:
#             char = item[1]  # 假设第二项是字
#             if char in standard_dict and pronunciation in standard_dict[char]:
#                 limited_values.append(item)

#     # 如果有有效条目，则保存到新数据中
#     if limited_values:
#         new_data[key] = limited_values

# # 将结果保存为 JSON 文件
# with open('data/train_data_v4.json', 'w', encoding='utf-8') as f:
#     json.dump(new_data, f, ensure_ascii=False, indent=4)

# print("数据处理完毕，已保存为 train_data_v4.json")

