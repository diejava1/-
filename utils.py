# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号


def build_vocab(file_path, tokenizer, max_size, min_freq):#创建字频表
    #文件路径   tokenizer不知道    总共最多我想要的字数      能出现在这个字典里面的最小的频率
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):#就是一个简单的进度条
            lin = line.strip()#去掉空格
            if not lin:
                continue
            content = lin.split('\t')[0]
            for word in tokenizer(content):
                vocab_dic[word] = vocab_dic.get(word, 0) + 1#找到就给字典这个+1，找不到就设置成1，比我用的if else 好多了
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset(config, ues_word):
    #config是初始化的组件    ues——word就是输入的究竟是词向量还是子向量
    if ues_word:#匿名函数作为参数传到别的函数里面
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        tokenizer = lambda x: [y for y in x]  # char-level
    if os.path.exists(config.vocab_path):#表示存在这个文件
        vocab = pkl.load(open(config.vocab_path, 'rb'))#打开这个文件
    else:#不存在
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))#保存此表
    print(f"Vocab size: {len(vocab)}")

    def load_dataset(path, pad_size=32):#加载训练集，验证集，测试集
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):#进度条
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')#一行前面的是内容，后面的是标签
                words_line = []
                token = tokenizer(content)#去掉空格
                seq_len = len(token)
                if pad_size:#每一句话都设置成一样的长度
                    if len(token) < pad_size:#比规定的长度短的话
                        token.extend([PAD] * (pad_size - len(token)))#用规定的符号来扩展成规定的长度
                    else:
                        token = token[:pad_size]#超过了规定长度，就截取前面的长度以内的部分
                        seq_len = pad_size
                # word to id
                for word in token:#对于集合里面的句子里面的每一个字，我都有一个对应的列表，我会把句子里面的每个字所对回应的字在字典里的频率都放进去，形成映射的关系
                    words_line.append(vocab.get(word, vocab.get(UNK)))#如果不存在这个字的话就放默认的符号的频率，但是不是不会出现这种情况吗，可见代码的严谨性
                contents.append((words_line, int(label), seq_len))#这个是把三元组放进列表里面
        return contents  # [([...], 0), ([...], 1), ...]
    #上面的例子应该有问题，应该是字频所组成的列表，加上标签的编号，加上句子的长度
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):#构造函数
        #batches就是所有的数据    btach——size就是一批数据的大小    device就是设别
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size#取整除法
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)#三元组的第一个，也就是子频组成的列表
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)#三元组的第二个，也就是句子的标签，也就是编号

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y    #我把字频的向量和长度的向量放在左边，变成一个二元组，把标签的向量放在右边

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]#获取最后剩下的一批用于带入模型的数据
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:#错误处理
            self.index = 0
            raise StopIteration#捕获错误
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]#获取新的一批用于带入模型的数据
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:#根据是不是正正好返回
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char.bz2"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)


