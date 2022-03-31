import numpy as np
import os
import json
def build_ent2query(data_dir):
    # 利用实体类型简介来描述 query
    ent2query = {
        '身份证号码': "ID_card",
        '其他组织': "other_org",
        '单位组织': "unit_org",
        '政府部门': "govern_org",
        '其他位置': "other_loc",
        '银行卡号': "bank_card",
        '团体': "group",
        '不确切时间': "incorrect_time",
        '钥匙': "keys",
        '其他事物': "other_things",
        '商业机构': "eco_org",
        '房产证': "house_property",
        '个体': "individual",
        '书籍条例':"books",
        '钱数':"money",
        'GPE':"GPE",
        '电话号码':"phone_num",
        '确切时间':"correct_time",
        '具体住址':"specific_loc",
        '其他人':"other_individuals",
        '项目产品':"projects",
        '合同':"contracts",
    }
    with open(os.path.join(data_dir, 'name_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(ent2query, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    build_ent2query('./')
    