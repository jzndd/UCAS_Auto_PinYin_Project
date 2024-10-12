import re
import json
from collections import defaultdict
from pypinyin import pinyin, Style
import jieba
from langchain_openai import ChatOpenAI
import warnings
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
from openai import OpenAI

import sys

sys.path.append('..')

warnings.filterwarnings("ignore")

SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.1/chat'
SPARKAI_APP_ID = 'your_app_id'
SPARKAI_API_SECRET = 'your_api_secret'
SPARKAI_API_KEY = 'your_api_key'
SPARKAI_DOMAIN = 'generalv3'

OpenAI_API_KEY = 'your_openai_api_key'

# 定义 extract_first_json 函数
def extract_first_json(text):
    """
    从给定的文本中提取第一个 JSON 数据。
    """
    # 尝试匹配被 ```json 和 ``` 包围的 JSON 块
    pattern_backticks = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(pattern_backticks, text)

    if match:
        json_str = match.group(1)
    else:
        # 如果未找到被反引号包裹的 JSON，尝试匹配第一个独立的 JSON 对象
        pattern_json = r'(\{[\s\S]*\})'
        match = re.search(pattern_json, text)
        if match:
            json_str = match.group(1)
        else:
            print("未找到符合条件的 JSON 数据。")
            return None

    # 打印提取到的 JSON 字符串以供调试
    print("提取到的 JSON 字符串（使用repr显示所有字符）：")
    #print(repr(json_str))

    # 尝试解析 JSON 数据
    try:
        json_obj = json.loads(json_str)
        return json_obj
    except json.JSONDecodeError as e:
        print(f"无效的 JSON 数据: {e}")
        # 尝试自动修复常见问题，例如尾随逗号
        json_str_fixed = json_str.replace("'", '"')
        json_str_fixed = re.sub(r',\s*([}\]])', r'\1', json_str_fixed)
        print("尝试修复后的 JSON 字符串：")
        print(repr(json_str_fixed))
        try:
            json_obj = json.loads(json_str_fixed)
            return json_obj
        except json.JSONDecodeError as e2:
            print(f"修复后仍然无效的 JSON 数据: {e2}")
            return None



# 定义 LLM API 调用函数
def call_llm_api(prompt):
    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=False,
    )
    messages = [ChatMessage(
        role="user",
        content=prompt
    )]
    handler = ChunkPrintHandler()
    a = spark.generate([messages], callbacks=[handler])
    return a.generations[0][0].text

def call_llm_api1(prompt):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        openai_api_key=OpenAI_API_KEY,  # 请在此处填写您的实际API密钥
        openai_api_base='https://api.cpdd666.cn/v1',
        temperature=0
    )
    a = model(prompt).content
    # print(a)
    return a


# 获取汉字的所有可能读音（包含声调）
def get_all_pronunciations(char):
    pronunciations = pinyin(char, style=Style.TONE3, heteronym=True)
    return pronunciations[0] if pronunciations else []


# 使用分词器提取包含指定字符且发音一致的词语及其位置
def extract_words_with_char(sentence, char, target_pronunciation):
    words = list(jieba.cut(sentence, cut_all=False, HMM=True))
    word_positions = []
    index = 0
    for word in words:
        word_len = len(word)
        if char in word:
            # 获取词语的拼音列表
            word_pinyins = pinyin(word, style=Style.TONE3)
            # 对齐字符和拼音
            for idx, (c, p) in enumerate(zip(word, word_pinyins)):
                if c == char and p[0] == target_pronunciation:
                    word_positions.append({
                        'word': word,
                        'start': index,
                        'end': index + word_len
                    })
                    break  # 找到匹配的发音，跳出内层循环
        index += word_len
    return word_positions, words


# 合并指定词语左右相邻的词语形成上下文
def get_context(words, target_index):
    context_words = []
    # 前一个词
    if target_index > 0:
        context_words.append(words[target_index - 1])
    # 当前词
    context_words.append(words[target_index])
    # 后一个词
    if target_index < len(words) - 1:
        context_words.append(words[target_index + 1])
    # 合并成上下文
    context = ''.join(context_words)
    return context


# 从文件中搜索多音字的例句
def find_examples(char, pronunciation):
    examples = []
    count = 0
    max_examples = 3

    # 读取文件1和文件2
    with open('file1.sent', 'r', encoding='utf-8') as f1, open('file2.lb', 'r', encoding='utf-8') as f2:
        file1_lines = f1.readlines()
        file2_lines = f2.readlines()
        # 遍历文件中的每一行
        for sentence_line, pinyin_line in zip(file1_lines, file2_lines):
            sentence_line = sentence_line.strip()
            pinyin_line = pinyin_line.strip()
            # 提取句子中的多音字及其位置
            positions = []
            for match in re.finditer(r'▁(.*?)▁', sentence_line):
                word = match.group(1)
                # 计算去除下划线后的实际位置
                start = match.start() - sentence_line[:match.start()].count('▁')
                end = start + len(word)
                positions.append({
                    'word': word,
                    'start': start,
                    'end': end
                })
            # 提取多音字的拼音列表
            pinyin_words = pinyin_line.replace('u:', 'ü').split()
            # 确保多音字和拼音数量一致
            if len(positions) != len(pinyin_words):
                print(f"警告：句子中的多音字数量与拼音数量不匹配。\n句子：{sentence_line}\n拼音：{pinyin_line}")
                continue
            # 去掉句子中的下划线
            clean_sentence = sentence_line.replace('▁', '')
            # 遍历句子中的多音字
            for idx, pos in enumerate(positions):
                word_char = pos['word']
                if char == word_char:
                    # 获取对应的拼音
                    word_pinyin = pinyin_words[idx]
                    if word_pinyin == pronunciation:
                        # 使用分词器提取包含多音字的词语及其位置
                        word_positions, words_list = extract_words_with_char(clean_sentence, char, pronunciation)
                        # 对每个包含多音字的词语，找到其在分词列表中的索引
                        index = 0
                        for i, w in enumerate(words_list):
                            if w in [wp['word'] for wp in word_positions]:
                                # 获取上下文
                                context = get_context(words_list, i)
                                examples.append({
                                    'sentence': clean_sentence,
                                    'word_context': context,
                                    'word': w
                                })
                                count += 1
                                if count >= max_examples:
                                    break
                        if count >= max_examples:
                            break
            if count >= max_examples:
                break
    return examples

def remove_punctuation(text):
    """去除中文和英文的标点符号"""
    import string
    chinese_punctuation = '。，“”‘’！？：；（）《》【】—……'
    punctuation = string.punctuation + chinese_punctuation
    translator = str.maketrans('', '', punctuation)
    return text.translate(translator)


def process_pinyin_dict(data_dict):
    """
    处理输入字典中的拼音字段，分割拼音音节并保留声调。

    参数:
        data_dict (dict): 输入的字典，必须包含 "pinyin" 键，其值为拼音字符串。

    返回:
        dict: 更新后的字典，"pinyin" 字段被分割为标准拼音音节并保留声调。
    """

    # 1. 定义所有标准拼音音节（不带声调）
    pinyin_syllables = [
        "a", "ai", "an", "ang", "ao",
        "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng", "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu",
        "ca", "cai", "can", "cang", "cao", "ce", "ceng", "cha", "chai", "chan", "chang", "chao", "che", "chen",
        "cheng", "chi", "chong", "chou", "chu", "chuai", "chuan", "chuang", "chui", "chun", "chuo",
        "ci", "cong", "cou", "cu", "cuan", "cui", "cun", "cuo",
        "da", "dai", "dan", "dang", "dao", "de", "deng", "di", "dian", "diao", "die", "ding", "diu", "dong", "dou",
        "du",
        "duan", "dui", "dun", "duo",
        "e", "en", "er",
        "fa", "fan", "fang", "fei", "fen", "feng", "fo", "fou", "fu",
        "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng", "gong", "gou", "gu", "gua", "guai", "guan",
        "guang", "gui", "gun", "guo",
        "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng", "hong", "hou", "hu", "hua", "huai", "huan",
        "huang", "hui", "hun", "huo",
        "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu", "ju", "juan", "jue", "jun",
        "ka", "kai", "kan", "kang", "kao", "ke", "ken", "keng", "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang",
        "kui", "kun", "kuo",
        "la", "lai", "lan", "lang", "lao", "le", "lei", "leng", "li", "lia", "lian", "liang", "liao", "lie", "lin",
        "ling", "liu", "long", "lou", "lu", "lv", "luan", "lue", "lun", "luo",
        "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng", "mi", "mian", "miao", "mie", "min", "ming",
        "miu", "mo", "mou", "mu",
        "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng", "ni", "nian", "niang", "niao", "nie", "nin",
        "ning", "niu", "nong", "nu", "nv", "nuan", "nue", "nuo",
        "o", "ou",
        "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng", "pi", "pian", "piao", "pie", "pin", "ping", "po",
        "pou", "pu",
        "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu", "qu", "quan", "que", "qun",
        "ran", "rang", "rao", "re", "ren", "reng", "ri", "rong", "rou", "ru", "ruan", "rui", "run", "ruo",
        "sa", "sai", "san", "sang", "sao", "se", "sen", "seng", "sha", "shai", "shan", "shang", "shao", "she",
        "shen", "sheng", "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun", "shuo",
        "si", "song", "sou", "su", "suan", "sui", "sun", "suo",
        "ta", "tai", "tan", "tang", "tao", "te", "teng", "ti", "tian", "tiao", "tie", "ting", "tong", "tou", "tu",
        "tuan", "tui", "tun", "tuo",
        "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu",
        "xi", "xia", "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu", "xu", "xuan", "xue", "xun",
        "ya", "yai", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you", "yu", "yuan", "yue",
        "yun",
        "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng", "zha", "zhai", "zhan", "zhang", "zhao",
        "zhe", "zhen", "zheng", "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui",
        "zhun", "zhuo",
        "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo"
        # 请确保包含所有拼音音节
    ]

    # 2. 创建一个函数去除拼音中的声调
    def strip_tone(pinyin):
        tone_marks = {
            'ā': 'a', 'á': 'a', 'ǎ': 'a', 'à': 'a',
            'ē': 'e', 'é': 'e', 'ě': 'e', 'è': 'e',
            'ī': 'i', 'í': 'i', 'ǐ': 'i', 'ì': 'i',
            'ō': 'o', 'ó': 'o', 'ǒ': 'o', 'ò': 'o',
            'ū': 'u', 'ú': 'u', 'ǔ': 'u', 'ù': 'u',
            'ǖ': 'ü', 'ǘ': 'ü', 'ǚ': 'ü', 'ǜ': 'ü',
            'ń': 'n', 'ň': 'n', '': 'm'
        }
        return ''.join(tone_marks.get(char, char) for char in pinyin)

    # 3. 编写分割函数，保留声调
    def split_pinyin_with_tone(pinyin_str, syllables):
        result = []

        # 预处理：移除空格
        pinyin_str = pinyin_str.replace(' ', '')

        i = 0
        length = len(pinyin_str)

        while i < length:
            match = None
            match_length = 0
            # 尝试匹配最长的音节
            for syllable in sorted(syllables, key=lambda x: len(x), reverse=True):
                syllable_length = len(syllable)
                if i + syllable_length > length:
                    continue
                # 提取当前部分
                current_part = pinyin_str[i:i + syllable_length]
                # 去除声调进行匹配
                current_part_stripped = strip_tone(current_part.lower())
                if current_part_stripped == syllable:
                    match = pinyin_str[i:i + syllable_length]
                    match_length = syllable_length
                    break
            if match:
                result.append(match)
                i += match_length
            else:
                # 如果没有匹配，保留原字符并前进
                result.append(pinyin_str[i])
                i += 1
        return ' '.join(result)

    # 检查输入字典是否包含 "pinyin" 键
    if "pinyin" not in data_dict:
        raise KeyError('输入字典必须包含 "pinyin" 键。')

    # 4. 提取并处理拼音字段
    original_pinyin = data_dict.get("pinyin", "")
    split_pinyin = split_pinyin_with_tone(original_pinyin, pinyin_syllables)

    # 5. 更新字典中的拼音字段
    data_dict["pinyin"] = split_pinyin

    return data_dict
def split_pinyin_in_dict(data):
    """
    接受一个包含拼音的字典，分割拼音中的多音节部分，同时保留声调。

    参数:
        data (dict): 输入的字典，格式如 {"sentence": "我爱你中国", "pinyin": "wǒ ài nǐ zhōngguó"}

    返回:
        dict: 经过拼音分割处理后的字典，格式如 {"sentence": "我爱你中国", "pinyin": "wǒ ài nǐ zhōng guó"}
    """
    if not isinstance(data, dict):
        raise TypeError("输入必须是一个字典")

    # 获取句子和原始拼音
    sentence = data.get('sentence', '')
    original_pinyin = data.get('pinyin', '')

    if not sentence:
        raise ValueError("字典中缺少 'sentence' 字段或字段为空")

    if not original_pinyin:
        raise ValueError("字典中缺少 'pinyin' 字段或字段为空")

    # 使用 pypinyin 将句子转换为拼音，保留声调
    pinyin_list = pinyin(sentence, style=Style.TONE, strict=False)

    # 将拼音列表扁平化为单个列表
    pinyin_flat = [item[0] for item in pinyin_list]

    # 将拼音列表连接成字符串，使用空格分隔
    split_pinyin_str = ' '.join(pinyin_flat)

    # 更新字典中的拼音字段
    data['pinyin'] = split_pinyin_str

    return data


def split_sentences(text):
    """
    将输入的文本按照句号、问号、感叹号等标点符号分割成句子，并保留标点符号。

    参数:
    text (str): 需要分割的文本。

    返回:
    list: 分割后的句子列表，每个句子末尾保留标点符号。
    """
    # 定义句子结束的标点符号
    sentence_endings = re.compile(r'([^。！？\n]+[。！？])')

    # 使用findall方法找到所有匹配的句子
    sentences = sentence_endings.findall(text)

    # 去除可能的空白字符
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return sentences


def load_json(file_path):
    """
    加载 JSON 数据

    :param file_path: JSON 文件路径
    :return: JSON 数据列表，如果加载失败则返回空列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的 JSON 格式。")
        return []

def is_char_present(data, character):
    """
    检测字符是否存在于 JSON 数据的 'char' 字段中

    :param data: JSON 数据列表
    :param character: 要检测的汉字
    :return: 如果存在返回 True，否则返回 False
    """
    for entry in data:
        if entry.get('char') == character:
            return True
    return False

def check_character(character):
    """
    封装函数：加载固定路径的 JSON 数据并检测指定的汉字是否存在

    :param character: 要检测的汉字
    :return: 如果存在返回 True，否则返回 False
    """
    # 固定的 JSON 文件路径
    json_file = 'data/polyphone.json'  # 请根据实际情况修改路径

    # 加载 JSON 数据
    data = load_json(json_file)

    if not data:
        print("无法加载 JSON 数据。")
        return False

    # 检测汉字是否存在
    return is_char_present(data, character)

def main():

    # 输入要处理的句子
    input_sentence = input("请输入要处理的句子：")
    # 获取句子中每个字的拼音（按句子中的读音，包含声调）

    # 修改 prompt1 以要求 JSON 格式回复
    prompt1 = (
        f"请仔细阅读这个句子：“{input_sentence}”，并根据语义将其进行分词分割，要特别注意“的”“地”这类词有可能跟在名词或动词后面，例如努力地应为一个词。\n"
        f"请以严格的 JSON 格式返回分词结果，使用双引号，格式如下：\n"
        f"{{\n"
        f"    \"segmented_sentence\": \"分词结果，用斜杠分隔\"\n"
        f"}}\n"
        f"请仅返回符合上述格式的 JSON 数据，不要附加任何解释。"
    )
    response1 = call_llm_api(prompt1)
    # print(response1)
    json1 = extract_first_json(response1)
    # print(1,json1)
    # if not json1 or "segmented_sentence" not in json1:
    #     print("无法解析分词结果。")
    #     print(f"LLM 返回内容：{response1}")  # 打印 LLM 返回内容以供调试
    #     return

    segmented_sentence = json1["segmented_sentence"]
    # print(segmented_sentence)
    # sentence_pinyin = pinyin(segmented_sentence, style=Style.TONE3)
    prompt = (
                    f"请阅读以下句子：“{segmented_sentence}”。\n"
                    f"请根据语义为每个分词注音，并将结果以严格的 JSON 格式返回，使用双引号。\n"
                    f"格式如下：\n"
                    f"{{\n"
                    f"    \"words\": [\n"
                    f"        {{\"word\": \"词语1\", \"pinyin\": \"拼音1\"}},\n"
                    f"        {{\"word\": \"词语2\", \"pinyin\": \"拼音2\"}},\n"
                    f"        ...\n"
                    f"    ]\n"
                    f"}}\n"
                    f"请仅返回符合上述格式的 JSON 数据，不要附加任何解释。"
                )
    response = call_llm_api(prompt)
    # print(response)
    json_response = extract_first_json(response)
    # print(json_response)
    words = json_response["words"]
    for item in words:
        if item['word'] =='地':
            print(1)
            item['pinyin'] = 'de'

    # print(2,words)
    # if not json_response or "words" not in json_response:
    #                 print("无法解析分词和注音结果。")
    #                 print(f"LLM 返回内容：{response}")  # 打印 LLM 返回内容以供调试
    # prompt2 = (
    #     f"你只需要做一个拼接的工人，而不需要自己创造或者更改，你要确保回复结果完全是我给你的输入的拼接。我会给你一句话和注音，你需要根据这句话和注音将结果拼凑出完整的一句话和注音, 任意两个拼音之间一定要空格隔开，无论这两个拼音是否属于同一分词。\n"
    #     f"请以严格的 JSON 格式返回结果，使用双引号，格式如下：\n"
    #     f"{{\n"
    #     f"    \"sentence\": \"完整句子\",\n"
    #     f"    \"pinyin\": \"完整句子的拼音\"\n"
    #     f"}}\n"
    #     f"例如：\n"
    #     f"{{\n"
    #     f"    \"sentence\": \"会计会算账\",\n"
    #     f"    \"pinyin\": \"hui4 ji4 hui4 suan4 zhang4\"\n"
    #     f"}}\n"
    #     f"下面请仔细阅读我给你的这句话：{input_sentence},分词注音:{words}"
    # )
    # response2 = call_llm_api(prompt2)
    # json2 = extract_first_json(response2)
    # if not json2 or "sentence" not in json2 or "pinyin" not in json2:
    #     print("无法解析完整句子和拼音。")
    #     print(f"LLM 返回内容：{response2}")  # 打印 LLM 返回内容以供调试
        # continue
    sentence = ''.join(item['word'] for item in words)
    pinyin = ' '.join(item['pinyin'] for item in words)
    json2 ={"sentence":sentence,"pinyin":pinyin}
    json2 = process_pinyin_dict(json2)
    print(json2)
    print(f"完整句子：{json2['sentence']}")
    print(f"完整拼音：{json2['pinyin']}")
    from pypinyin import pinyin as pypinyin_func, Style

    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'

    annotated_sentence = ''

    sentence = json2['sentence']
    pinyin_str = json2['pinyin']
    sentence_clean = remove_punctuation(sentence)
    if len(sentence_clean)!= len(remove_punctuation(input_sentence)):
        print(input_sentence)
        prompt1 = (
            f"请仔细阅读这个句子：“{input_sentence}”，并根据语义将其进行分词分割，要特别注意“的”“地”这类词有可能跟在名词或动词后面，例如努力地应为一个词。\n"
            f"请以严格的 JSON 格式返回分词结果，使用双引号，格式如下：\n"
            f"{{\n"
            f"    \"segmented_sentence\": \"分词结果，用斜杠分隔\"\n"
            f"}}\n"
            f"请仅返回符合上述格式的 JSON 数据，不要附加任何解释。"
        )
        response1 = call_llm_api1(prompt1)
        # print(response1)
        json1 = extract_first_json(response1)
        segmented_sentence = json1["segmented_sentence"]
        # print(segmented_sentence)
        # sentence_pinyin = pinyin(segmented_sentence, style=Style.TONE3)
        prompt = (
            f"请阅读以下句子：“{segmented_sentence}”。\n"
            f"请根据语义为每个分词注音，并将结果以严格的 JSON 格式返回，使用双引号。\n"
            f"格式如下：\n"
            f"{{\n"
            f"    \"words\": [\n"
            f"        {{\"word\": \"词语1\", \"pinyin\": \"拼音1\"}},\n"
            f"        {{\"word\": \"词语2\", \"pinyin\": \"拼音2\"}},\n"
            f"        ...\n"
            f"    ]\n"
            f"}}\n"
            f"请仅返回符合上述格式的 JSON 数据，不要附加任何解释。"
        )
        response = call_llm_api1(prompt)
        # print(response)
        json_response = extract_first_json(response)
        # print(json_response)
        words = json_response["words"]
        for item in words:
            if item['word'] == '地':
                #print(1)
                item['pinyin'] = 'de'
        sentence = ''.join(item['word'] for item in words)
        pinyin = ' '.join(item['pinyin'] for item in words)
        json2 = {"sentence": sentence, "pinyin": pinyin}
        json2 = process_pinyin_dict(json2)
        #print(json2)
        print(f"完整句子：{json2['sentence']}")
        print(f"完整拼音：{json2['pinyin']}")
        from pypinyin import pinyin as pypinyin_func, Style

        def is_chinese(char):
            return '\u4e00' <= char <= '\u9fff'

        annotated_sentence = ''

        sentence = json2['sentence']
        pinyin_str = json2['pinyin']
    pinyin_clean = remove_punctuation(pinyin_str)
    pinyin_list = pinyin_clean.strip().split()

    if len([c for c in sentence if is_chinese(c)]) != len(pinyin_list):
        print("警告：句子中的汉字数量和拼音数量不一致。")
        # 需要对齐汉字和拼音，可能需要更复杂的处理
    else:
        pinyin_idx = 0
        for char in input_sentence:
            if is_chinese(char):
                pinyin_char = pinyin_list[pinyin_idx]
                pinyin_idx += 1

                if check_character(char):
                    # 这是多音字，注音
                    annotated_sentence += f"{char}({pinyin_char})"
                else:
                    annotated_sentence += char
            else:
                # 非中文字符，直接添加
                annotated_sentence += char

    print(f"注音后的句子：{annotated_sentence}")


if __name__ == "__main__":
    main()
