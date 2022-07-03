import sys
import jieba
from 分词 import *
from PyQt5.QtWidgets import *
from HMM import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt

sign=False


def fmm(text):
    fmm_result = fmm(text)
    string = 'FMM:'
    for i in fmm_result:
        if i == fmm_result[0]:
            string = string + i
        else:
            string = string + '/' + i
    print(string)


def bmm(text):
    bmm_result = bmm(text)
    string = 'BMM:'
    for i in bmm_result:
        if i == bmm_result[0]:
            string = string + i
        else:
            string = string + '/' + i
    print(string)


def hmm(text):
    if sign == False:
        init_model()
        train('result1.pkl')
    hmm_result = cut(text, 1)  # hmm 里面的 cut
    string = 'HMM:'
    for i in hmm_result:
        if i == hmm_result[0]:
            string = string + i
        else:
            string = string + '/' + i
    print(string)


def jb(text):

    jb_result = list(jieba.cut(text))
    string = 'Jieba:'
    for i in jb_result:
        if i == jb_result[0]:  # 第一项
            string = string + i
        else:  # 不是第一项
            string = string + '/' + i
    print(string)



