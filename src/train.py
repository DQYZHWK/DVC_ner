from itertools import zip_longest
from copy import deepcopy
from dvclive import Live
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
from box import ConfigBox
from ruamel.yaml import YAML
from util import tensorized, sort_by_lengths, cal_loss, cal_lstm_crf_loss
import time
import json
import pickle
yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        # 词嵌入层：将输入的词索引映射为词向量
        
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)
        # 2*hidden_size 表示双向LSTM输出的特征维度的两倍（因为双向LSTM的输出包含正向和反向两个方向的隐藏状态
        self.lin = nn.Linear(2*hidden_size, out_size)
        self.num = 0
    def forward(self, sents_tensor, lengths):
        self.num +=1
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]

        # pack_padded_sequence 的作用是将填充后的序列重新打包（pack）为一个紧凑的序列，以减少冗余计算。
        # 这个函数在处理带有填充的序列时非常有用，特别是在使用 LSTM、GRU 等循环神经网络进行处理时。
        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        rnn_out, _ = self.bilstm(packed)
        # rnn_out:[B, L, hidden_size*2]
        if self.num == 1:
            print("sents_tensor: ")# [64,179]
            print(sents_tensor.shape)
            print("emb: ")# [64,179,128]
            print(emb.shape)
            print("packed")
            #print(packed.shape)
            print("1: rnn_out :")
            #print(rnn_out.shape)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        scores = self.lin(rnn_out)  # [B, L, out_size]
        if self.num ==1:
            print("2: rnn_out :")# [64,179,256]
            print(rnn_out.shape)
            print("scores: ")# [64,179,32]
            print(scores.shape)
        return scores

    def test(self, sents_tensor, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM_CRF, self).__init__()
        self.bilstm = BiLSTM(vocab_size, emb_size, hidden_size, out_size)

        # CRF实际上就是多学习一个转移矩阵 [out_size, out_size] 初始化为均匀分布
        # nn.Parameter()函数创建了一个可学习的参数张量（tensor），并将其标记为模型的参数。这意味着在训练过程中，优化器会自动更新这些参数，以最小化损失函数。
        self.transition = nn.Parameter(
            torch.ones(out_size, out_size) * 1/out_size)
        # self.transition.data.zero_()
        self.num = 0

    def forward(self, sents_tensor, lengths):
        # [B, L, out_size]
        emission = self.bilstm(sents_tensor, lengths)
        self.num+=1
        # 计算CRF scores, 这个scores大小为[B, L, out_size, out_size]
        # 也就是每个字对应对应一个 [out_size, out_size]的矩阵
        # 这个矩阵第i行第j列的元素的含义是：上一时刻tag为i，这一时刻tag为j的分数
        batch_size, max_len, out_size = emission.size()
        # tenser.unsqueeze(2),在第2维度增加一个向量
        crf_scores = emission.unsqueeze(
            2).expand(-1, -1, out_size, -1) + self.transition.unsqueeze(0)
        if self.num ==1:
            print(crf_scores.shape)
        return crf_scores

    def test(self, test_sents_tensor, lengths, tag2id):
        """使用维特比算法进行解码"""
        start_id = tag2id['<start>']
        end_id = tag2id['<end>']
        pad = tag2id['<pad>']
        tagset_size = len(tag2id)

        crf_scores = self.forward(test_sents_tensor, lengths)
        device = crf_scores.device
        # B:batch_size, L:max_len, T:target set size
        B, L, T, _ = crf_scores.size()
        # viterbi[i, j, k]表示第i个句子，第j个字对应第k个标记的最大分数
        viterbi = torch.zeros(B, L, T).to(device)
        # backpointer[i, j, k]表示第i个句子，第j个字对应第k个标记时前一个标记的id，用于回溯
        backpointer = (torch.zeros(B, L, T).long() * end_id).to(device)
        lengths = torch.LongTensor(lengths).to(device)
        # 向前递推
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                # 第一个字它的前一个标记只能是start_id
                viterbi[:batch_size_t, step,
                        :] = crf_scores[: batch_size_t, step, start_id, :]
                backpointer[: batch_size_t, step, :] = start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    crf_scores[:batch_size_t, step, :, :],     # [B, T, T]
                    dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags

        # 在回溯的时候我们只需要用到backpointer矩阵
        backpointer = backpointer.view(B, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(L-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == L-1:
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += end_id
            else:
                prev_batch_size_t = len(tags_t)

                new_in_batch = torch.LongTensor(
                    [end_id] * (batch_size_t - prev_batch_size_t)).to(device)
                offset = torch.cat(
                    [tags_t, new_in_batch],
                    dim=0
                )  # 这个offset实际上就是前一时刻的
                index = torch.ones(batch_size_t).long() * (step * tagset_size)
                index = index.to(device)
                index += offset.long()

            try:
                tags_t = backpointer[:batch_size_t].gather(
                    dim=1, index=index.unsqueeze(1).long())
            except RuntimeError:
                import pdb
                pdb.set_trace()
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())

        # tagids:[L-1]（L-1是因为扣去了end_token),大小的liebiao
        # 其中列表内的元素是该batch在该时刻的标记
        # 下面修正其顺序，并将维度转换为 [B, L]
        tagids = list(zip_longest(*reversed(tagids), fillvalue=pad))
        tagids = torch.Tensor(tagids).long()

        # 返回解码的结果
        return tagids


class BILSTM_Model(object):
    def __init__(self, vocab_size, out_size):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        
        self.emb_size = params.lstm.emb_size
        self.hidden_size = params.lstm.hidden_size

        self.crf = params.lstm.if_crf
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if self.crf == 0:
            self.model = BiLSTM(vocab_size, self.emb_size,
                                self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_loss
        else:
            self.model = BiLSTM_CRF(vocab_size, self.emb_size,
                                    self.hidden_size, out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss

        # 加载训练参数：
        self.epoches = params.training.epoches
        self.print_step = params.training.print_step
        self.lr = params.training.lr
        self.batch_size = params.training.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self, word_lists, tag_lists,
              dev_word_lists, dev_tag_lists,
              word2id, tag2id):

        torch.manual_seed(params.training.random_seed)
        torch.cuda.manual_seed_all(params.training.random_seed)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样

        # 对数据集按照长度进行排序 在后序tensor化时实现对齐
        word_lists, tag_lists, _ = sort_by_lengths(word_lists, tag_lists)
        dev_word_lists, dev_tag_lists, _ = sort_by_lengths(
            dev_word_lists, dev_tag_lists)

        B = self.batch_size
        with Live("results/train", report=None,save_dvc_exp=True) as live:
            live.log_params(yaml.load(open("params.yaml", encoding="utf-8")))
            for e in range(1, self.epoches+1):
                self.step = 0
                losses = 0.
                epoch_losses = 0
                for ind in range(0, len(word_lists), B):
                    # batch_size个小的list
                    batch_sents = word_lists[ind:ind+B]
                    batch_tags = tag_lists[ind:ind+B]
                    loss = self.train_step(batch_sents,batch_tags, word2id, tag2id)
                   
                    losses += loss
                    epoch_losses +=loss
                    if self.step % params.training.print_step == 0:
                        total_step = (len(word_lists) // B + 1)
                        print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                            e, self.step, total_step,
                            100. * self.step / total_step,
                            losses / self.print_step
                        ))
                        losses = 0.
                live.log_metric('train/loss',epoch_losses/self.step)
                # 每轮结束测试在验证集上的性能，保存最好的一个
                val_loss = self.validate(
                    dev_word_lists, dev_tag_lists, word2id, tag2id)
                live.log_metric('eval/loss',val_loss)
                print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))
                
                live.next_step()
                
            # 训练结束，把整个Bilstm_Model对象存贮下来    
            save_model(self,"models/model.pkl")
            live.log_artifact("models/model.pkl", type="model", name="lstm or lstm_crf model")

    def train_step(self, batch_sents, batch_tags, word2id, tag2id):
        self.model.train()
        self.step += 1
        # 准备数据
        # 每个输入tensor是在字典中的下标，如果不足补充PAD，如果不存在换成UNK
        # lengths 一个batch里每个小单元的长度
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        targets, lengths = tensorized(batch_tags, tag2id)
        targets = targets.to(self.device)

        # forward
        scores = self.model(tensorized_sents, lengths)

        # 计算损失 更新参数
        # pytorch 中 如果出现tensor to tensor 那么就会加入计算图
        self.optimizer.zero_grad()
        loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, tag_lists, word2id, tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        with Live("results/evaluate", report=None, cache_images=True,save_dvc_exp=True) as live:
            word_lists, tag_lists, indices = sort_by_lengths(word_lists, tag_lists)
            tensorized_sents, lengths = tensorized(word_lists, word2id)
            tensorized_sents = tensorized_sents.to(self.device)

            self.best_model.eval()
            with torch.no_grad():
                batch_tagids = self.best_model.test(
                    tensorized_sents, lengths, tag2id)

            # 将id转化为标注
            pred_tag_lists = []
            id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
            for i, ids in enumerate(batch_tagids):
                tag_list = []
                if self.crf:
                    for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                        tag_list.append(id2tag[ids[j].item()])
                else:
                    for j in range(lengths[i]):
                        tag_list.append(id2tag[ids[j].item()])
                pred_tag_lists.append(tag_list)

            # indices存有根据长度排序后的索引映射的信息
            # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
            # 索引为2的元素映射到新的索引是1...
            # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
            ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
            indices, _ = list(zip(*ind_maps))
            pred_tag_lists = [pred_tag_lists[i] for i in indices]
            tag_lists = [tag_lists[i] for i in indices]
            
            preds=[]
            reals=[]
            for outer in pred_tag_lists:
                for inter in outer:
                    preds.append(inter)
            for outer in tag_lists:
                for inter in outer:
                    reals.append(inter)
            print(reals)    
            live.log_sklearn_plot("confusion_matrix",reals, preds, name="cm.json")

def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

def load_lists_from_json(filename):
    with open(f'data/processed/{filename}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    word_lists, tag_lists = zip(*data)
    return word_lists, tag_lists
   
def train():

    with open('data/processed/word2id.json', 'r', encoding='utf-8') as f:
        word2id = json.load(f)

    with open('data/processed/tag2id.json', 'r', encoding='utf-8') as f:
        tag2id = json.load(f)
    
    train_word_lists, train_tag_lists = load_lists_from_json('train_data')
    dev_word_lists, dev_tag_lists = load_lists_from_json('dev_data')
    
    # word_lists 为两重vector，为[[1条数据的中文字列表],[],[...]]
    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)
    bilstm_model = BILSTM_Model(vocab_size, out_size)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)
    
    #save_model(bilstm_model,"models/model.pkl")
    print("训练完毕,共用时{}秒.".format(int(time.time()-start)))


if __name__ =='__main__':
    train()