import os
import json
from tqdm import trange
import numpy as np
from sklearn.model_selection import train_test_split, KFold

filtration = ["诉求","事实描述","抱怨","格式语句","银行贷款还不上","政府越界_房","抵押问题","一般行为","物业管理问题","非法放贷","负面行为","开发商恶劣行为","逾期交房","房屋建设问题","对业主进行诈骗","ORG","TIME","开发商虚假承诺","其他信息","LOC","政府越界_金","INFO","政府渎职_房","政府渎职_金","银行问题服务","开发商涉黑","存款丢失","个人诈骗","非法集资"]
with open(os.path.join('./', 'name_dict.json'), encoding='utf-8') as f:
    ent2id = json.load(f)

##将data写入data_dir文件夹下desc.json文件
def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2) ##写入文件 ##ensure_ascii允许非ascii字符


def convert_data_to_json(base_dir, save_data=False, save_dict=False):
    stack_examples = []

    stack_dir = os.path.join(base_dir, 'train')

    # process train examples
    for i in trange(4877): ##trange会打印进度条
        with open(os.path.join(stack_dir, f'xfj_{i:04d}.txt'), encoding='utf-8') as f:
            text = f.read().strip()

        labels = []
        with open(os.path.join(stack_dir, f'xfj_{i:04d}.ann'), encoding='utf-8') as f:
            for line in f.readlines():
                tmp_label = line.strip().split('\t')
                if (tmp_label[0][0] == "E"):
                    continue
                assert len(tmp_label) == 3
                tmp_mid = tmp_label[1].split() ##应该分成三段：实体类型，开始位置，结束位置
                if (tmp_mid[0] in filtration):
                    continue
                tmp_mid[0] = ent2id[tmp_mid[0]]
                tmp_label = [tmp_label[0]] + tmp_mid + [tmp_label[2]] ##长度为5

                labels.append(tmp_label)
                tmp_label[2] = int(tmp_label[2]) ##开始位置
                tmp_label[3] = int(tmp_label[3]) ##结束位置

                assert text[tmp_label[2]:tmp_label[3]] == tmp_label[-1], '{},{}索引抽取错误'.format(tmp_label, i)

        stack_examples.append({'id': i,
                               'text': text,
                               'labels': labels,
                               'pseudo': 0}) ##是否为test_set


    # 构建实体知识库
    kf = KFold(10)
    entities = set() ##存所有的文本内容
    ent_types = set() ##存所有的实体类型
    for _now_id, _candidate_id in kf.split(stack_examples): #Generate indices to split data into training and test set
        now = [stack_examples[_id] for _id in _now_id] ##训练集
        candidate = [stack_examples[_id] for _id in _candidate_id] ##测试集
        now_entities = set() ##存当前训练集的所有文本内容

        for _ex in now: ##迭代9/10的训练集
            for _label in _ex['labels']: ##_label有五行：Ti,实体类型，开始位置，结束位置,文本内容
                ent_types.add(_label[1]) ##实体类型

                if len(_label[-1]) > 1:
                    now_entities.add(_label[-1])
                    entities.add(_label[-1])
        # print(len(now_entities))
        for _ex in candidate: ##迭代1/10的测试集
            text = _ex['text']
            candidate_entities = []

            for _ent in now_entities:
                if _ent in text: ##该文本内容出现在该条测试数据的text中
                    candidate_entities.append(_ent)

            _ex['candidate_entities'] = candidate_entities ##该测试集里面拥有的文本内容
    assert len(ent_types) == 22 ##总共22种实体类型

    train, dev = train_test_split(stack_examples, shuffle=True, random_state=123, test_size=0.15)

    if save_data:
        save_info(base_dir, stack_examples, 'stack')
        save_info(base_dir, train, 'train')
        save_info(base_dir, dev, 'dev')

    if save_dict:
        ent_types = list(ent_types)
        span_ent2id = {_type: i+1 for i, _type in enumerate(ent_types)}

        ent_types = ['O'] + [p + '-' + _type for p in ['B', 'I', 'E', 'S'] for _type in list(ent_types)]
        crf_ent2id = {ent: i for i, ent in enumerate(ent_types)} ##crf模型要弄成这样

        mid_data_dir = os.path.join(os.path.split(base_dir)[0], 'mid_data')
        if not os.path.exists(mid_data_dir):
            os.mkdir(mid_data_dir)

        save_info(mid_data_dir, span_ent2id, 'span_ent2id')
        save_info(mid_data_dir, crf_ent2id, 'crf_ent2id')


def build_ent2query(data_dir):
    # 利用实体类型简介来描述 query
    ent2query = {
        'ID_card': "数字号码",
        'other_org': "不确定的组织",
        'unit_org': "法院等组织",
        'govern_org': "政府直属部门",
        'other_loc': "不是具体位置的地址",
        'bank_card': "数字号码",
        'group': "大于1人",
        'incorrect_time': "不准确时间",
        'keys': "钥匙",
        'other_things': "其他事物",
        'eco_org': "商业机构",
        'house_property': "房产证",
        'individual': "个体",
        'books':"书籍条例",
        'money':"钱数",
        'GPE':"GPE",
        'phone_num':"电话号码",
        'correct_time':"确切时间",
        'specific_loc':"具体住址",
        'other_individuals':"其他人",
        'projects':"项目产品",
        'contracts':"合同",
    }

    with open(os.path.join(data_dir, 'mrc_ent2id.json'), 'w', encoding='utf-8') as f:
        json.dump(ent2query, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    convert_data_to_json('../../raw_data', save_data=True, save_dict=True)
    build_ent2query('../../mid_data')

