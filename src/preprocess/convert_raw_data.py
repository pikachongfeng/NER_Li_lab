import os
import json
from tqdm import trange
import numpy as np
from sklearn.model_selection import train_test_split, KFold

filtration = ["LOC","INFO","其他事物","其他信息","ORG","负面行为","TIME","其他人","其他组织","一般行为"]
with open(os.path.join('./', 'name_dict.json'), encoding='utf-8') as f:
    ent2id = json.load(f)

##将data写入data_dir文件夹下desc.json文件
def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2) ##写入文件 ##ensure_ascii允许非ascii字符


def convert_data_to_json(base_dir, file_path,save_data=False, save_dict=False):
    stack_examples = []

    with open(file_path, encoding='utf-8') as f:
        raw_examples = json.load(f)

    for i, item in enumerate(raw_examples):
        text = item["doc_text"]
        labels = []
        sentences = item["sentences"]
        for _, sentence in enumerate(sentences):
            entities = sentence["entities"]
            for _, entity in enumerate(entities):
                type = entity["entity_type"]
                if type in filtration:
                    continue
                type = ent2id[type]
                id  = entity["entity_id"]
                idx = entity["entity_idx"]
                start = idx[0]
                end = idx[1] 
                entity_text = entity["entity_text"]
                tmp_label = [id] + [type] + [start] + [end] + [entity_text]
                assert entity_text == text[start:end], '{},{}索引抽取错误'.format(tmp_label, i)
                labels.append(tmp_label)

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
    assert len(ent_types) == 19 ##总共19种实体类型

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
        'unit_org': "法院等组织",
        'govern_org': "政府直属部门",
        'other_loc': "不是具体位置的地址",
        'bank_card': "数字号码",
        'group': "大于1人",
        'incorrect_time': "不准确时间",
        'keys': "钥匙",
        'eco_org': "商业机构",
        'house_property': "房产证",
        'individual': "个体",
        'books':"书籍条例",
        'money':"钱数",
        'GPE':"GPE",
        'phone_num':"电话号码",
        'correct_time':"确切时间",
        'specific_loc':"具体住址",
        'projects':"项目产品",
        'contracts':"合同",
    }

    with open(os.path.join(data_dir, 'mrc_ent2id.json'), 'w', encoding='utf-8') as f:
        json.dump(ent2query, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    convert_data_to_json('../../raw_data', '../../dierzu.json',save_data=True, save_dict=True)
    build_ent2query('../../mid_data')

