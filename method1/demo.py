
from docx import Document
from pypinyin import pinyin, Style

# read the docx file
def read_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Convert Chinese characters to pinyin
def add_pinyin(text):
    pinyin_text = pinyin(text, style=Style.TONE3)  
    return [''.join(word) for word in pinyin_text]     

docx_file_path = 'test.docx'  # 替换为你的 .docx 文件路径
text = read_docx(docx_file_path)
print("原文：\n", text)

pinyin_list = add_pinyin(text)
pinyin_text = ' '.join(pinyin_list)
print("注音版：\n", pinyin_text)