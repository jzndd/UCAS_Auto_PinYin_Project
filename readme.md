## 代码结构

data 存放**模型文件**（.pth, 需要去我的云盘下载）、**训练数据** （.json）、**测试文档与输出文档** （.docx）以及 pypinyin 的支持库 （polyphone.json）  

method1   

|_ method1.py 运行，直接读取docx 文档并注音

method2  

|_ get_data_v_.py 获取各种类型数据集的文件，活比较脏，没必要看，处理思路会写在文档里  

|_ train.py 训练模型，获得 pth 文件  

|_ test.py 加载模型，推理，输入 docx 文件，输出 docx 推理后的文件

|_ nlp_nan.py 处理难字

## TODO:

LLM 方式作为 method3 合并到代码中  

- [ ] LLM Teacher-Student  @qjk

- [ ] LLM 语义理解          @hr

method2:  

- [x] 模型文件 v4 的上传 @jzn  

- [x] 模型推理时，一个句子中有重复的多音字会出现 bug @jzn 

UI
 
- [ ] UI 合并到仓库中 @znx @sl

benchmark :

- [x] 一个常规的文档（nlp_test.docx） , 一个极端样例 （郝锐群里的“干一行爱一行等”），需要量化 @？

