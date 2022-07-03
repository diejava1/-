import pickle


f = open('result1.pkl', 'rb')
dictionary = pickle.load(f)
maximum = 19


# 检验是否全是中文字符
def is_all_chinese(this_str):
    for _char in this_str:
        if not '\u4e00' <= _char <= '\u9fa5':
            return 0
    return 1


def bmm(text):
    result = []
    index = len(text)#文章的末尾
    sign = -2
    th = ''#index每次指向的都是截取的片段的最后一个的编号+1
    while index > 0:
        word = None
        for size in range(maximum, 0, -1):#每次都是尽可能取出最长的
            if index - size < 0:
                continue
            piece = text[(index - size):index]#从句子里截取一段长度为size的子句
            if piece in dictionary:
                word = piece
                result.append(word)
                index -= size
                break
        if word is None:#所有的长度都尝试过了，然后都没有这个词语，所以这个字单独构成一个
            if is_all_chinese(text[index-1]) and index >= 1:
                result.append(text[index-1])
                index -= 1
            else:#如果这个是标点符号
                if sign == index+1:
                    th += text[index-1]#加上这个
                    if index == 1 or is_all_chinese(text[index-2]):#走到了最前面
                        result.append(th[::-1])#反向塞进去
                        th = ''
                    sign = index
                    index -= 1
                else:
                    th += text[index-1]
                    if index == 1 or is_all_chinese(text[index-2]):#如果到头了或者是这个标点符号的前面就是汉字，而不是连着的标点符号
                        result.append(th[::-1])
                        th = ''
                    sign = index
                    index -= 1
    return result[::-1]#把结果再转向




def fmm(text):#从前往后
    result = []
    index = 0
    sign = -2
    th = ''
    while index < len(text):
        word = None
        for size in range(maximum, 0, -1):#每次匹配尽可能大的
            if index + size > len(text):
                continue
            piece = text[index:index+size]
            if piece in dictionary:
                word = piece
                result.append(word)
                index += size
                break
        if word is None:
            if is_all_chinese(text[index]):
                result.append(text[index])
                index += 1
            else:
                if sign == index-1:
                    th += text[index]
                    if index + 1 == len(text) or is_all_chinese(text[index]):
                        result.append(th)
                        th = ''
                    sign = index
                    index += 1
                else:
                    th += text[index]
                    if index + 1 == len(text) or is_all_chinese(text[index]):
                        result.append(th)
                        th = ''
                    sign = index
                    index += 1
    return result
