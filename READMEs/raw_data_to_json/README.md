序列标注任务通常是用brat网站，最后对每个文本会得到一个ann（标注结果）和一个txt（原始文档）文件，一般会先把这类文件转换成json



convert_raw_data.py提供了将ann和txt转换成json文件的功能，并对bert-crf, bert-span, bert-mrc三种模型生成了在构建输入时需要使用的字典