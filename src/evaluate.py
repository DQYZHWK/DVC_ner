from train import BILSTM_Model,BiLSTM,BiLSTM_CRF
import pickle
import json
import time
from box import ConfigBox
from ruamel.yaml import YAML
yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))



def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

def load_lists_from_json(filename):
    with open(f'data/processed/{filename}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    word_lists, tag_lists = zip(*data)
    return word_lists, tag_lists

def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model

def evaluate():
    # 这里的字典在data阶段已经经过处理了
    with open('data/processed/word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)

    with open('data/processed/tag2id.json', 'r', encoding='utf-8') as f:
        tag2id = json.load(f)
    
    test_word_lists, test_tag_lists = load_lists_from_json('test_data')
    
    start = time.time()
    print("加载并评估模型...")
    model_path = 'models/model.pkl'
    bilstm_model = load_model(model_path)
    #bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    
    bilstm_model.test(test_word_lists, test_tag_lists,word2id, tag2id)

    print("评估完毕,共用时{}秒.".format(int(time.time()-start)))
    
if __name__ == '__main__':
    evaluate()