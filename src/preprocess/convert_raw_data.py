import os
import json
from tqdm import trange
import numpy as np
from sklearn.model_selection import train_test_split, KFold

with open(os.path.join('./', 'name_dict.json'), encoding='utf-8') as f:
    ent2id = json.load(f)

##将data写入data_dir文件夹下desc.json文件
def save_info(data_dir, data, desc):
    with open(os.path.join(data_dir, f'{desc}.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2) ##写入文件 ##ensure_ascii允许非ascii字符


def convert_data_to_json(base_dir, save_data=False, save_dict=False):
    stack_examples = []

    stack_dir = os.path.join(base_dir, 'train')

    f = open("../../source.txt","r") 
    f2 = open("../../target.txt","r")
    lines = f.readlines()
    lines2 = f2.readlines()
    i = 0
    for line in lines:
        text = line.replace(' ','').strip()

        labels=[]
        line2 = lines2[i].strip().split(" ")
        lineList = line.strip().split(" ")
        start = 0
        j = 0
        while j < len(line2):
            label = line2[j]
            id = "T" + str(j)
            if label != 'O':
                entity = label[2:]
                word = lineList[j]
                start_idx = start
                j += 1                    
                while (j < len(line2) and len(line2[j]) > 0 and line2[j][0] == 'I' and line2[j][2:] == entity):
                    word += lineList[j]
                    j += 1
                end_idx = start + len(word)
                tmp_label = [id] + [entity] + [start_idx] + [end_idx] + [word]
                labels.append(tmp_label)
                start = end_idx
            else:
                start = len(lineList[j]) + start
                j += 1
                
        stack_examples.append({'id': i,
                               'text': text,
                               'labels': labels,
                               'pseudo': 0}) ##是否为test_set
        i += 1

    f.close()
    f2.close()

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
    assert len(ent_types) == 7,'{}'.format(ent_types) ##总共7种实体类型

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
        'TIME': "时间",
        'ORG': "组织",
        'ROLE': "职业",
        'PER': "人物",
        'LOC': "地点",
        'CRIME': "罪犯",
        'LAW': "法律",
    }

    with open(os.path.join(data_dir, 'mrc_ent2id.json'), 'w', encoding='utf-8') as f:
        json.dump(ent2query, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    convert_data_to_json('../../raw_data', save_data=True, save_dict=True)
    build_ent2query('../../mid_data')

