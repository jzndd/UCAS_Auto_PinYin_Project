import json

# 定义文件路径
input_file_path = r'D:\Meng\UCAS\\nlp\UCAS_Auto_PinYin_Project\method1\\rare_char.txt'  # 替换为你的TXT文件路径
output_file_path = 'rare_char.json'  # 输出的JSON文件路径

# 初始化一个列表用于存储数据
data = []

# 读取TXT文件
with open(input_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 分割每行数据
        parts = line.strip().split('\t')
        
        # 创建字典并添加到列表
        if len(parts) >= 6:
            entry = {
                'index': int(parts[0]),
                'char': parts[1],
                'frequency': int(parts[2]),
                'score': float(parts[3]),
                'pinyin': parts[4].split('/'),

            }
            data.append(entry)

# 将数据写入JSON文件
with open(output_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)

print(f"数据已成功转换并保存为 {output_file_path}")
