import json
from os.path import join
from codecs import open
from box import ConfigBox
from ruamel.yaml import YAML
yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

def build_corpus(split, make_vocab=True, data_dir="data/raw"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    with open(join(data_dir, split+".char.bmes"), 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        return word_lists, tag_lists, word2id, tag2id
    else:
        return word_lists, tag_lists

# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=1):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


# 这里start存在的必要性
def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists
def save_lists_to_json(word_lists, tag_lists, filename):
    data = list(zip(word_lists, tag_lists))
    with open(f'data/processed/{filename}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def data():
    print("读取数据...")
    # 处理raw数据
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf = params.lstm.if_crf)
    
        
    # 如果是加了CRF，数据集还需要额外的一些数据处理
    if params.lstm.if_crf:
        train_word_lists, train_tag_lists = prepocess_data_for_lstmcrf(
            train_word_lists, train_tag_lists
        )
        dev_word_lists, dev_tag_lists = prepocess_data_for_lstmcrf(
            dev_word_lists, dev_tag_lists
        )
        test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
            test_word_lists, test_tag_lists, test=True
        )
    # 列表套列表，每个列表单元最后有一个<end>
    #print(test_word_lists)


    with open('data/processed/word2id.json', 'w', encoding='utf-8') as f:
        json.dump(crf_word2id, f, ensure_ascii=False)

    with open('data/processed/tag2id.json', 'w', encoding='utf-8') as f:
        json.dump(crf_tag2id, f, ensure_ascii=False)

    save_lists_to_json(train_word_lists, train_tag_lists, 'train_data')

    save_lists_to_json(dev_word_lists, dev_tag_lists, 'dev_data')

    save_lists_to_json(test_word_lists, test_tag_lists, 'test_data')

if __name__ == "__main__":
    data()