读取json格式文件，构建bert-crf, bert-span, bert-mrc三种模型的输入



# BERT_base_feature

```python
class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids):
        # BERT 输入
        # BertTokenizer.encode_plus返回的
        self.token_ids = token_ids #token在字典中对应id
        self.attention_masks = attention_masks #token是否遮盖
        self.token_type_ids = token_type_ids #token对应的句子id
```

# BERT-crf

```python
class CRFFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None,
                 pseudo=None,
                 distant_labels=None):
        super(CRFFeature, self).__init__(token_ids=token_ids,
							attention_masks=attention_masks,                               								token_type_ids=token_type_ids)
        # labels, 每个token的label，包括['S','B','E','I'] + ['-'] + [entity_type]
        self.labels = labels
        # pseudo，是否是测试集做的pseudo训练集(半监督学习)
        self.pseudo = pseudo
        # 远程监督 label，在构建训练集的时候，用交叉验证的办法，如果某个实体类型对应的文本内容同时出现在训练集和测试集中，记录为candidate_entities, 最后会作为distant_labels
        self.distant_labels = distant_labels
```



![6](/Users/zhangyangrong/python_learning/NLP/NER/md_files/6.png)



# BERT-span

融合模型1——BERT-SPAN

- 采用SPAN指针的形式替代CRF模块，加快训练速度
- 以半指针-半标注的结构预测实体的起始位置，同时标注过程中给出实体类别
- 采用严格解码形式，重叠实体选取logits最大的一个，保证准确率
- 使用label smooth缓解过拟合问题

![10](/Users/zhangyangrong/python_learning/NLP/NER/md_files/10.png)

# BERT-mrc

融合模型2——BERT-MRC

- 基于阅读理解的方式处理NER任务
  - query：实体类型的描述来作为query
  - doc：分句后的原始文本作为doc
- 针对每一种类型构造一个样本，训练时有大量负样本，可以随机选取30%加入训练，其余丢弃，保证效率
- 预测时对每一类都需构造一次样本，对解码输出不做限制，保证召回率
- 使用label smooth缓解过拟合问题
- MRC在本次数据集上精度表现不佳，且训练和推理效率较低，仅作为提升召回率的方案，提供代码仅供学习，不推荐日常使用

![11](/Users/zhangyangrong/python_learning/NLP/NER/md_files/11.png)



# 多级模型融合

多级融合策略

- CRF/SPAN/MRC 5折交叉验证得到的模型进行第一级概率融合，将 logits 平均后解码实体
- CRF/SPAN/MRC 概率融合后的模型进行第二级投票融合，获取最终结果![12](/Users/zhangyangrong/python_learning/NLP/NER/md_files/12.png)