import pickle

# 状态值集合:B-词首 M-词中 E-词尾 S-单独的词
state_list = ['B', 'M', 'E', 'S']

#长度大于1的就是B
#长度等于1的就是S
#统计的只是个数


# 状态转移概率:状态->状态
transform = dict()

# 发射概率:字->状态
emit = dict()#每个状态都放好一个字典，每个字典里面放的就是摸一个字出现的次数

# 状态初始概率
origin = dict()#统计词语和独立的汉字的数量

# 状态出现次数
count_s = dict()#统计每个状态的数量


# 初始化
def init_model():
    for i in state_list:
        # 初始化转移、初始概率、出现次数，均赋值为0
        # 发射概率设为空dict
        transform[i] = {s: 0.0 for s in state_list}
        origin[i] = 0
        count_s[i] = 0
        emit[i] = dict()


# 词->状态
def label(input_text):
    output1 = []#放的是对应的label
    output2 = []#放的是词语或者是汉字
    if len(input_text) == 0:
        return [], []
    if len(input_text) == 1:#如果只是一个单独的汉字
        output1.append('S')
        output2.append(input_text)
        if input_text == '。':
            # 句子用空格划分
            output1.append(' ')
            output2.append(' ')
    else:
        output1 += ['B'] + ['M'] * (len(input_text) - 2) + ['E']#-2就是词语除了词首和词尾
        for i in input_text:
            output2.append(i)#把这个词语中的一个个汉字放进去
    return output1, output2


# 训练HMM
def train(path):
    global transform, emit, origin, count_s, state_list
    f = open(path, 'rb')
    dictionary = pickle.load(f)
    # 建立word_list,为了计算发射概率
    word_list = []
    # 计算初始概率
    first = pickle.load(f)
    f.close()#把所有的分出来的词语都放在first表里面
    for i in first:
        # 每个句子的第一个字状态为B or S
        if len(i) > 1:
            origin['B'] += 1
        elif len(i) == 1:
            origin['S'] += 1
        else:
            print('Warning!:' + i)
    # 字及标点等的状态
    word_state = []
    # 将训练语料库中的词转换为状态
    for w in dictionary:
        t1, t2 = label(w)
        word_state.extend(t1)
        word_list.extend(t2)
    # 设置断言
    assert len(word_state) == len(word_list)#检验是不是真的相等
    # 计算
    for k, v in enumerate(word_state):
        if v == ' ':
            continue
        else:
            # 统计状态次数
            count_s[v] += 1
            if word_state[k - 1] != ' ':
                # 转移+1
                transform[word_state[k - 1]][v] += 1#表示从某一个状态到达另外一个状态的次数有多少
            else:
                # 转移+1
                transform[word_state[k - 2]][v] += 1
            # 发射+1
            emit[word_state[k]][word_list[k]] = emit[word_state[k]].get(word_list[k], 0.0) + 1.0#统计这个字出现在几种状态的次数
    # 计算初始概率
    origin = {k: float(v) / len(first) for k, v in origin.items()}
    # 计算转移概率
    transform = {k: {k1: float(v1) / count_s[k] for k1, v1 in v.items()}
                 for k, v in transform.items()}#计算转移过去的这个数量占总的数量的比例
    # 计算发射概率(加法平滑)
    emit = {k: {k1: float(v1 + 1) / count_s[k] for k1, v1 in v.items()}
            for k, v in emit.items()}#记录这个字在这个状态的比例是多少
    # save
    # with open('p.pkl', 'wb') as f1:
    #     pickle.dump(transform, f1)
    #     pickle.dump(emit, f1)
    #     pickle.dump(origin, f1)


# 动态规划求解最大概率路径
def max_path(text, sign):
    global transform, emit, origin, state_list
    v = [{}]
    #对于每一个字，字对应编号，都要放一个字典，字典的key是每一个标签类型，值是对应的概率，就是每个字下一次变成各个状态的概率
    path = {}
    # 第一个字的概率
    for i in state_list:
        v[0][i] = origin[i] * emit[i].get(text[0], 0)#找到对应的这个启始的标签*这个起始的标签的前提之下的这个第一个字的条件概率
        path[i] = [i]
    # 第一个字之后
    for j in range(1, len(text)):
        v.append({})
        next_path = {}
        # bool:没有这个字的发射概率
        none_w = False
        if text[j] not in emit['S'].keys() and text[j] not in emit['M'].keys() and text[j] not in emit['E'].keys() and text[j] not in emit['B'].keys():
            none_w = True
            print('None Word')
            print(text[j])
        # 每一个字遍历各种状态
        for k in state_list:
            # 未知字作为一个词，发射概率1.0。已知，则正常get
            #从k状态发射到这个现在被读到的字的概率
            emit_P = emit[k].get(text[j], 0) if not none_w else 1.0
            # 概率p，状态state
            # 从前一个字生成后一个字的最大概率=max(前一个字的概率*状态转化概率*转换后状态发射到后一个字的概率)
            (p, state) = max(
                [(v[j - 1][y] * transform[y].get(k, 0) * emit_P, y) for y in state_list if v[j - 1][y] > 0])#对于前者的所有大于0的这个标签的概率*各个状态转换这到个现在
            #被遍历到的这个状态的概率*
            v[j][k] = p
            next_path[k] = path[state] + [k]
        path = next_path
    if sign == 0:
        (p, state) = max([(v[len(text) - 1][y], y) for y in state_list])
    else:
        (p, state) = max([(v[len(text) - 1][y], y) for y in ['E', 'S']])
        if p == 0:
            (p, state) = max([(v[len(text) - 1][y], y) for y in state_list])
    return p, path[state]


# 分词
def cut(text, sign):
    start, follow = 0, 0
    fc = []
    p_list, path = max_path(text, sign)
    for i, w in enumerate(text):
        p = path[i]
        if p == 'B':
            start = i
        elif p == 'E':
            fc.append(text[start:i + 1])#每次读到标签为词尾的
            follow = i + 1#记录读到哪了，可以把最后剩下的给输出来
        elif p == 'S':
            fc.append(w)
            follow = i + 1
    if follow < len(text):
        fc.append(text[follow:])
    return fc




