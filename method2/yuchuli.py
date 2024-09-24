import re

# 打开文件
file_path = '199801.txt'  # 请确保文件路径正确
with open(file_path, 'r', encoding='utf-8') as file:
    # 读取文件内容
    content = file.read()

def clean_text(text):
    # Remove date-like strings at the start of the line
    text = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}/m ', '', text, flags=re.MULTILINE)

    # Remove all Arabic numbers
    text = re.sub(r'\d+', '', text)

    # Remove all book title marks and quotes
    text = re.sub(r'《|》|"|“|”', '', text)

    # Split the text by lines
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Split each line by space, then split each element by '/' and take the first part
        words = [word.split('/')[0] for word in line.split() if '/' in word]

        # Join the words and fix extra spaces before punctuation
        cleaned_line = ' '.join(words)
        cleaned_line = re.sub(r'\s+([，。！？、])', r'\1', cleaned_line)

        cleaned_lines.append(cleaned_line)

    # Join the cleaned lines back into a single text
    cleaned_text = '\n'.join(cleaned_lines)

    # Remove all spaces
    cleaned_text = cleaned_text.replace(" ", "")

    return cleaned_text

def further_refined_clean_text(text):
    # Remove date-like strings at the start of the line
    text = re.sub(r'^\d{8}-\d{2}-\d{3}-\d{3}/m ', '', text, flags=re.MULTILINE)

    # Remove all Arabic numbers
    text = re.sub(r'\d+', '', text)

    # Remove all book title marks, quotes, and additional specified characters
    text = re.sub(r'《|》|"|“|”|\'|‘|’|\(|\)|\[|\]|%|\.', '', text)

    # Keep only specified Chinese punctuation
    text = re.sub(r'[^\u4e00-\u9fa5，。！？、]', '', text)

    return text

# Clean the text
cleaned_text = further_refined_clean_text(clean_text(content))

# 然后，将清理后的文本写入到 'wenben.txt' 文件中
output_file_path = 'wenben.txt'

# 写入文件
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write(cleaned_text)
