
import numpy as np
import torch.nn
from openTSNE import TSNE
import matplotlib.pyplot as plt

do_end_to_end = True
do_feature_ensemble = True
do_grad_ensemble = True

do_output_ensemble = False
do_loss_ensemble = False

batch_evaluate = False

def CalcHammingDist(code1, code2):
    q = code2.shape[1]   # bits
    distH = 0.5 * (q - np.dot(code1, code2.transpose()))  # 矩阵乘法
    return distH

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
def find_s_e(database_code, database_label, one_hot_labels, n_class):
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
def get_center(database_code, database_label, n_class, bit, s, e):
    # print(s, e)
    center = []
    for i in range(n_class):  # 28类
        tmp = []
        for j in range(bit):  # 每类300个code, 投票出一个center, 要对64bit分别投票
            pos = 0  # +1
            neg = 0  # -1
            for k in range(s[i], e[i] + 1):  # 第k个code的第j位是1 or -1
                if database_code[k][j] == 1:
                    pos += 1
                else:
                    neg += 1
            # if pos >= neg:
            #     tmp.append(1)
            # else:
            #     tmp.append(-1)
            tmp.append(float((pos - neg) / (pos + neg)))
        center.append(tmp)

    centers_np = np.zeros((n_class, bit), dtype='float')
    for i in range(n_class):  # 100个code
        # 数组转numpy形式, 然后np.save
        center_i = np.array(center[i]).astype(float)
        centers_np[i] = center_i

    # print(centers_np)
    return centers_np

def train_attacker():
    org_codes_ = np.load('/data/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/database_code.npy')
    codes1_ = np.load('/data/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/database_code.npy')
    codes2_ = np.load('/data/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/database_code.npy')
    codes3_ = np.load('/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/database_code.npy')

    label = np.load('/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/database_label.npy')

    # print(np.sum(org_codes != codes16))    # 15886/12370
    # return

    # centers = np.concatenate((org_codes, codes12))
    # centers = TSNE(n_components=2).fit(centers)
    # plt.scatter(centers[0:len(org_codes), 0], centers[0:len(org_codes), 1], color='blue', label='set1', marker='.', alpha=0.5,
    #             s=80)
    # plt.scatter(centers[len(org_codes):, 0], centers[len(org_codes):, 1], color='orange', label='set2', marker='.',
    #             alpha=0.5, s=80)
    #
    # plt.legend(loc='best')
    # plt.savefig('./codes_csq/two_sets')
    # plt.show()
    # return


    n_class, bit = 28, 64
    one_hot_labels = get_label(n_class)
    s, e = find_s_e(org_codes_, label, one_hot_labels, n_class)
    ### 总体数据分布不太相同, 稍有偏差, 但中心的中心一致(直接所有图像投得的中心还是不太一致的)
    org_codes = get_center(org_codes_, label, n_class, bit, s, e)
    codes1 = get_center(codes1_, label, n_class, bit, s, e)
    codes2 = get_center(codes2_, label, n_class, bit, s, e)
    codes3 = get_center(codes3_, label, n_class, bit, s, e)

    # print(np.sum(org_codes[0] * codes1[0] > 0))     # 方向都一致, 数值上有差异 45/64
    # return

    length_db = len(org_codes_)
    print(org_codes_.shape)
    from utils.votingForCenter import voting_center, voting_anchors
    org_centers = voting_anchors(org_codes, num_spts=0, hash_bit=64, is_father=True).numpy()
    codes1_center = voting_anchors(codes1, num_spts=0, hash_bit=64, is_father=True).numpy()
    codes2_center = voting_anchors(codes2, num_spts=0, hash_bit=64, is_father=True).numpy()
    codes3_center = voting_anchors(codes3, num_spts=0, hash_bit=64, is_father=True).numpy()

    # print(np.sum(codes2_center != codes10_center))
    # return


    print(codes1_center.shape)
    org_centers = org_centers.reshape(1, -1)
    codes1_center = codes1_center.reshape(1, -1)
    codes2_center = codes2_center.reshape(1, -1)
    codes3_center = codes3_center.reshape(1, -1)

    print(codes1_center.shape)
    # print(np.sum(codes7_center*org_centers))

    centers = np.concatenate((org_centers, codes1_center))
    centers = np.concatenate((centers, codes2_center))
    centers = np.concatenate((centers, codes3_center))
    print(centers.shape)

    # print(org_codes_.shape, codes1_.shape, codes2_.shape, codes3_.shape)
    centers = np.concatenate((centers, org_codes_))
    print(centers.shape)

    centers = np.concatenate((centers, codes1_))
    print(centers.shape)

    centers = np.concatenate((centers, codes2_))
    print(centers.shape)

    centers = np.concatenate((centers, codes3_))
    print(centers.shape)


    # centers = np.concatenate((centers, org_codes_))
    # print(centers.shape)

    # min_dist, max_dist = 99999, -1
    # min_i, min_j, max_i, max_j = -1, -1, -1, -1
    # for i, ci in enumerate(centers):
    #     for j, cj in enumerate(centers):
    #         ci = ci.reshape(1, -1)
    #         cj = cj.reshape(1, -1)
    #         if i == j:
    #             continue
    #         dist = CalcHammingDist(ci, cj)
    #         # l2 = torch.nn.MSELoss()
    #         # dist = l2(torch.from_numpy(ci), torch.from_numpy(cj))
    #         print(i, j, dist)
    #         if dist < min_dist:
    #             min_dist = dist
    #             min_i = i
    #             min_j = j
    #         if dist > max_dist:
    #             max_dist = dist
    #             max_i = i
    #             max_j = j
    #
    # print('max_dist:', max_dist, ', max_i, max_j:', max_i, max_j)
    # print('min_dist:', min_dist, ', min_i, min_j:', min_i, min_j)
    #
    # return
    centers = TSNE(n_components=2).fit(centers)

    # plt.scatter(centers[17:, 0], centers[17:, 1], color='green', label='org_set',
    #             marker='.', alpha=0.5, s=80)

    # plt.scatter(centers[0, 0], centers[0, 1], color='red', marker='*', alpha=0.5,
    #             s=80)
    # plt.scatter(centers[1, 0], centers[1, 1], color='red', marker='*',
    #             alpha=0.5, s=80)
    # plt.scatter(centers[2, 0], centers[2, 1], color='red', marker='*',
    #             alpha=0.5, s=80)
    # plt.scatter(centers[3, 0], centers[3, 1], color='red', marker='*',
    #             alpha=0.5, s=80)
    plt.scatter(centers[4:length_db+4, 0], centers[4:length_db+4, 1], color='#6AB4C1', label='ResNet34', marker='.',
                alpha=0.5, s=50)
    plt.scatter(centers[length_db+4:2*length_db+4, 0], centers[length_db+4:2*length_db+4, 1], color='#FF7F00',
                label='ResNet50', marker='.', alpha=0.5, s=50)
    plt.scatter(centers[2*length_db+4:3*length_db+4, 0], centers[2*length_db+4:3*length_db+4, 1], color='#70B48F', label='VGG16', marker='.',
                alpha=0.5, s=50)
    plt.scatter(centers[3*length_db+4:4*length_db+4, 0], centers[3*length_db+4:4*length_db+4, 1], color='#5861AC',
                label='VGG19', marker='.', alpha=0.5, s=50)
    plt.scatter(centers[0, 0], centers[0, 1], color='#ED5151', marker='*', alpha=0.5,
                s=200, edgecolor='#EA3232')
    plt.scatter(centers[1, 0], centers[1, 1], color='#ED5151', marker='*',
                alpha=0.5, s=200, edgecolor='#EA3232')
    plt.scatter(centers[2, 0], centers[2, 1], color='#ED5151', marker='*',
                alpha=0.5, s=200, edgecolor='#EA3232')
    plt.scatter(centers[3, 0], centers[3, 1], color='#ED5151', marker='*',
                alpha=0.5, s=200, edgecolor='#EA3232')


    plt.legend(loc='upper right')
    plt.savefig('./tsne/4clusters')
    plt.show()


if __name__ == "__main__":
    train_attacker()
