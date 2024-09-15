import os
import numpy as np

train_n = 50
test_n = train_n + 100

if __name__ == '__main__':
    # label
    one_hot_labels = []
    for i in range(28):
        one_hot_label = np.zeros(28, dtype=int)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    path = 'CASIA-WebFace'
    dirs = os.listdir(path)
    dirs = sorted(dirs)
    num_dir = len(dirs)
    with open('data/CASIA/train.txt', 'w') as f1:
        with open('data/CASIA/test.txt', 'w') as f2:
            with open('data/CASIA/database.txt', 'w') as f3:
                n = 0
                for dir_ in dirs:
                    path_ = path + '/' + dir_
                    files = os.listdir(path_)
                    files = sorted(files)
                    num_file = len(files)
                    if num_file >= 500:
                        cnt = 0
                        for file in files:
                            pth = path_ + '/' + file
                            path_and_label = pth + ' ' + str(one_hot_labels[n])[1:len(str(one_hot_labels[n])) - 1]
                            path_and_label = path_and_label.replace('\n', '')
                            cnt += 1
                            if cnt <= train_n:
                                f1.write(path_and_label + "\n")
                                print(path_and_label)
                            elif cnt <= test_n:
                                f2.write(path_and_label + "\n")
                                print(path_and_label)
                            else:
                                f3.write(path_and_label + "\n")
                                print(path_and_label)
                        n += 1