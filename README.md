# NER_Li_lab

## 数据挖掘

![文本长度分布直方图](https://github.com/pikachongfeng/NER_Li_lab/blob/main/data_mining/%E6%96%87%E6%9C%AC%E9%95%BF%E5%BA%A6%E5%88%86%E5%B8%83%E7%9B%B4%E6%96%B9%E5%9B%BE.png)

部分文本过长，考虑过滤掉



![不同标签样本数](https://github.com/pikachongfeng/NER_Li_lab/blob/main/data_mining/不同标签样本数.png)

标签分布不均衡，考虑在模型层面解决此问题



## 过滤部分实体

文本过长类实体，如负面行为、一般行为

错标，如TIME,INFO

不合理实体，如其他人、其他事物（主观判定不合理）



## 保留19个实体

'身份证号码': "ID_card",

​        '其他组织': "other_org",

​        '单位组织': "unit_org",

​        '政府部门': "govern_org",

​        '其他位置': "other_loc",

​        '银行卡号': "bank_card",

​        '团体': "group",

​        '不确切时间': "incorrect_time",

​        '钥匙': "keys",

​        '其他事物': "other_things",

​        '商业机构': "eco_org",

​        '房产证': "house_property",

​        '个体': "individual",

​        '书籍条例':"books",

​        '钱数':"money",

​        'GPE':"GPE",

​        '电话号码':"phone_num",

​        '确切时间':"correct_time",

​        '具体住址':"specific_loc",

​        '其他人':"other_individuals",

​        '项目产品':"projects",

​        '合同':"contracts",



## 训练集、测试集分布

9:1



## CRF, SPAN, MRC模型

##### BERT-CRF模型训练 

```python
task_type='crf'
mode='train' or 'stack'  train:单模训练与验证 ； stack:5折训练与验证

swa_start: swa 模型权重平均开始的 epoch
attack_train： 'pgd' / 'fgm' / '' 对抗训练 fgm 训练速度慢一倍, pgd 慢两倍，pgd 本次数据集效果明显
```

##### BERT-SPAN模型训练 

```python
task_type='span'
mode：同上
attack_train: 同上
loss_type: 'ce'：交叉熵; 'ls_ce'：label_smooth; 'focal': focal loss
```

##### BERT-MRC模型训练 

```python
task_type='mrc'
mode：同上
attack_train: 同上
loss_type: 同上
```

修改run.sh里的参数，然后运行即可跑不同的模型、不同的参数、加上不同的layer（如attck_train)

显卡好的windows电脑或者云主机可以直接跑下试试，不然就在Colab上跑

每个模型都可以跑一下，最后看结果编故事，BERT-CRF模型已经不用再跑了。
