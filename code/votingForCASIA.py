import numpy as np
import os

# one hot label初始化
def get_label(n_class):
    one_hot_labels = []
    for i in range(n_class):
        one_hot_label = np.zeros(n_class, dtype=int)
        one_hot_label[i] = 1
        one_hot_label = str(one_hot_label)
        end = len(one_hot_label) - 1
        one_hot_labels.append(one_hot_label[1:end])
    return one_hot_labels

# 每类有num个label, 找到开始行数和结束行数(划分类, 以类为单位再生成hashcenter)
def find_s_e(database_code, database_label, one_hot_labels):
    s, e = [], []

    for j in range(n_class):
        s_, e_ = 0, 0
        flag = 0
        for i in range(database_code.shape[0]):
            # print(one_hot_labels[j], str(database_label[i]))
            if one_hot_labels[j] in str(database_label[i]):
                if flag == 0:
                    s_ = i
                    flag = 1
                if i == len(database_code) - 1 or one_hot_labels[j] not in str(database_label[i + 1]):
                    e_ = i
                    s.append(s_)
                    e.append(e_)
                    break
    return s, e

# 得到center numpy格式
def get_center(database_code, database_label, n_class, bit):
    one_hot_labels = get_label(n_class)
    s, e = find_s_e(database_code, database_label, one_hot_labels)
    
    for i in range(n_class):   # 28类
        tmp = []
        for j in range(bit):   # 每类300个code, 投票出一个center, 要对64bit分别投票
            pos = 0  # +1
            neg = 0  # -1
            for k in range(s[i], e[i] + 1):  # 第k个code的第j位是1 or -1
                if database_code[k][j] == 1:
                    pos += 1
                else:
                    neg += 1
            if pos >= neg:
                tmp.append(1)
            else:
                tmp.append(-1)
        center.append(tmp)

    centers_np = np.zeros((n_class, bit), dtype='float')
    for i in range(n_class):    # 100个code
        # 数组转numpy形式, 然后np.save
        center_i = np.array(center[i]).astype(float)
        centers_np[i] = center_i
        
    print(centers_np)
    return centers_np

if __name__ == '__main__':
    # 用database_code投票算出每类的center code
    save_path = './save/CSQ/Vgg11/CASIA/0.851461589082956/'
    database_code = np.load(save_path + 'database_code.npy')  # code
    database_label = np.load(save_path + 'database_label.npy')    # label
    n_class = 28   # 28类
    bit = 64       # hash bit   最后要得到28个64位的center
    center = []
    
    centers_np = get_center(database_code, database_label, n_class, bit)
    np.save(os.path.join(save_path, 'hashcenters.npy'), centers_np)