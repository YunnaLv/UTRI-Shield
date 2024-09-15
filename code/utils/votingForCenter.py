import torch
import numpy as np
import random


# index -> hashcenters[index]
def get_org_index(list_path):
    lines = open(list_path).readlines()
    num = len(lines)
    index = []
    for i in range(num):
        label = lines[i].replace('\n', '').split(' ')[1:]
        index_ = label.index('1')
        index.append(index_)
    return set(index)


def voting_center(list_path, hashcenters, hash_bit):
    index = get_org_index(list_path)
    pos = [0 for i in range(hash_bit)]
    neg = [0 for i in range(hash_bit)]
    for ind in index:
        center = torch.from_numpy(hashcenters[ind]).unsqueeze(0)
        for bit in range(hash_bit):
            if center[0][bit] == 1:
                pos[bit] += 1
            else:  # -1
                neg[bit] += 1
    target = [0 for i in range(hash_bit)]
    for bit in range(hash_bit):
        if pos[bit] >= neg[bit]:
            target[bit] = 1
        else:
            target[bit] = -1
    target = torch.tensor(target).to(torch.float32)
    return target


def voting(inds, hashcenters, hash_bit):
    pos = [0 for _ in range(hash_bit)]
    neg = [0 for _ in range(hash_bit)]
    for ind in inds:
        center = torch.from_numpy(hashcenters[ind]).unsqueeze(0)
        for bit in range(hash_bit):
            if center[0][bit] == 1:
                pos[bit] += 1
            else:  # -1
                neg[bit] += 1
    target = [0 for _ in range(hash_bit)]
    for bit in range(hash_bit):
        if pos[bit] >= neg[bit]:
            target[bit] = 1
        else:
            target[bit] = -1
    target = torch.tensor(target).to(torch.float32)
    return target


def voting_anchors(hashcenters, num_spts, hash_bit, is_father=False, min_idx=3):
    max_anchors = len(hashcenters)
    if is_father:
        inds = [i for i in range(max_anchors)]
        anchor = voting(inds, hashcenters, hash_bit)
        return anchor
    else:
        anchor_sets = []
        for j in range(num_spts):
            rand_num_of_classes = np.random.randint(2, max_anchors)
            inds = random.sample(range(min_idx, max_anchors), min(rand_num_of_classes, max_anchors - min_idx))
            sub_anchor = voting(inds, hashcenters, hash_bit)
            anchor_sets.append(sub_anchor)
        return np.stack(anchor_sets)
