# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from train_eval import test
from importlib import import_module#动态导入对象
from train_eval import MyTest
from train_eval import MyTest2


def ceshi(a):
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'

    model_name = a  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)  # x就是导入的这个模型，所以可以调用这个模型x的里面的函数
    config = x.Config(dataset, embedding)  # 预先铺好路
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()  # 开始时间
    print("Loading data...")
    # 创建初始的一些东西
    vocab, train_data, dev_data, test_data = build_dataset(config, False)  # 创建交叉验证集
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    test(config, model, test_iter)



