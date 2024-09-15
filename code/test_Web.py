import torch
from network import Vgg, ResNet
from utils.tools import ImageList, CalcTopMap, image_transform, CalcHammingDist
from utils.noise_utils import *
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from DiffJPEG import DiffJPEG
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAP_robust/model/torch-model'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


cpu_num = 5     # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# 模型加载
def load_model(hash_bit, specific_model, model_path):
    if 'ResNet' in specific_model:
        model = ResNet(hash_bit, specific_model)  # hash_bit and specific model type
    elif 'Vgg' in specific_model:
        model = Vgg(hash_bit, specific_model)

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.eval()
    return model.cuda()


# 图像变换
def image_transform(resize_size, crop_size, data, dataset):  # data(train, test, database), dataset(CASIA..)
    if data == "train":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]  # 测试集

    # mean和std由computeMeanAndStd.py计算而来
    if dataset == 'people':
        mean_ = [0.474, 0.433, 0.418]
        std_ = [0.309, 0.296, 0.295]
    elif dataset == 'CASIA':
        mean_ = [0.496, 0.385, 0.324]
        std_ = [0.284, 0.245, 0.236]
    elif dataset == 'imagenet':  # imagenet-50...
        mean_ = [0.485, 0.456, 0.406]
        std_ = [0.229, 0.224, 0.225]
    elif dataset == 'vggfaces2':
        mean_, std_ = [0.596, 0.456, 0.390], [0.263, 0.228, 0.219]

    return transforms.Compose(
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=mean_, std=std_)]
                              )

# 数据加载(预处理 + loader)
def load_data(data_path, list, batch_size, resize_size, crop_size, data, dset):
    # dataset = ImageList(data_path, open(list_path).readlines(),
    #                     transform=image_transform(resize_size, crop_size, data, dset))
    dataset = ImageList(data_path, list, transform=image_transform(resize_size, crop_size, data, dset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

# 压缩
def image_jpeg(images, quality, dataset):
    jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=quality)
    return jpeg(images, dataset)

# img + noise 计算net(dataloader + noise)结果
def compute_result(dataloader, noise, img_save_path, net, device, dataset, QF):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        perturbated_images = clamp_img(img + noise, dataset)
        # org   不进入if else
        if QF != None:
            perturbated_images = image_jpeg(perturbated_images, QF, dataset)
            img = image_jpeg(img, QF, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        if img_save_path != None:
            save_image(un_normalize(perturbated_images[0], dataset), img_save_path)
        bs.append((net(perturbated_images.to(device))).data.cpu())  # 扰动图像
        bs_2.append((net(img.to(device))).data.cpu())  # 原图
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)

def save_pert(dataloader, noise, img_save_path, img_name, dataset, QF):
    cnt = 0
    for img, cls, _ in tqdm(dataloader):
        perturbated_images = clamp_img(img + noise, dataset)
        # org   不进入if else
        if QF != None:
            perturbated_images = image_jpeg(perturbated_images, QF, dataset)
            img = image_jpeg(img, QF, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)

        save_image(un_normalize(perturbated_images[0], dataset), img_save_path + img_name[cnt])
        cnt += 1

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

### 让dbCode和path相关(返回排序index)
def CalcTopMap(dbCode, queryCode, dbLabel, queryLabel, topk):
    # print(dbCode.shape, queryCode.shape, dbLabel, queryLabel, topk)
    num_query = queryLabel.shape[0]
    topk_map = 0
    index = []
    hammingdist = []
    crt = []
    for iter in range(num_query):
        same = (np.dot(queryLabel[iter, :], dbLabel.transpose()) > 0).astype(int)   # label是不是一致(是不是检索正确)
        hammdist = CalcHammingDist(queryCode[iter, :], dbCode)   # 计算query与database所有图像的Hamming距离
        ind = np.argsort(hammdist)  # 汉明距离排序, 得到新的排序的[索引]
        index.append(ind[:topk])
        hammingdist.append(hammdist[ind[:topk]])
        crt.append(same[ind[:topk]])
        same = same[ind]
        topk_same = same[0:topk]    # topK 是否检索正确
        # print(iter, 'topk_same', topk_same)
        topk_sum = np.sum(topk_same).astype(int)  # 一共几个检索正确的
        # print(topk_sum)
        if topk_sum == 0:
            continue
        count = np.linspace(1, topk_sum, topk_sum)  # 创建数组, [1, 2, ..., topk_sum]   分子

        topk_index = np.asarray(np.where(topk_same == 1)) + 1.0  # top几的时候检索正确(0开始 所以要加1)   分母
        # print('.....', topk_index, count)
        topk_map_ = np.mean(count / (topk_index))        # 计算对于这个query的mAP
        # print('---', topk_map_)
        topk_map = topk_map + topk_map_
    topk_map = topk_map / num_query
    return topk_map, index, hammingdist, crt

def test_noise_single(model_save_path, img_save_path_org, img_save_path_adv, label, model, noise, QF):
    database_code = np.load(model_save_path + "database_code.npy")
    database_label = np.load(model_save_path + "database_label.npy")
    # 加载database code + label   (检索算法中计算出来的)
    database_txt_path = './data/CASIA/database.txt'
    database_img_path = np.array(open(database_txt_path).readlines())
    # 数据集
    dset = "CASIA"
    # 加载测试集
    data_path = ''
    # list_path = './data/CASIA/database.txt'
    list = [img_save_path_org + ' ' + label]
    # print('test_list =', list)
    test_loader = load_data(data_path, list, 1, 255, 224, 'test', dset)
    noise = torch.from_numpy(noise)
    noise = clamp_noise(noise, dset)
    topk = 30
    per_codes, org_codes, org_labels = compute_result(test_loader, noise, img_save_path_adv, model, device="cuda:0", dataset=dset, QF=QF)  # tqdm
    org_mAP, org_index, org_dist, _ = CalcTopMap(database_code, org_codes, database_label, org_labels, topk)
    per_mAP, per_index, per_dist, _ = CalcTopMap(database_code, per_codes, database_label, org_labels, topk)
    result_path, result_label, result_dist = '', '', ''
    print(per_mAP, org_mAP)
    database_path = open(database_txt_path).readlines()
    per_index = per_index[0].tolist()

    for i in range(len(per_index)):
        # print(i, per_index[i])
        res_path = '/data/UTAH_datasets/CASIA-WebFace/' + database_path[per_index[i]].split(' ')[0]
        res_label_ = database_path[per_index[i]].split(' ')[1:]
        res_label = ''
        for j in range(len(res_label_)):
            res_label += res_label_[j]
        res_label = res_label.replace('\n', '')
        res_dist = per_dist[0][i]
        result_path = result_path + str(res_path) + '|'
        result_label = result_label + str(res_label) + '|'
        result_dist = result_dist + str(res_dist) + '|'

    return org_mAP, per_mAP, result_path, result_label, result_dist     # (result = path, label, dist)

# list_path, img_save_path, noise, QF
def test_noise_multi(model_save_path, list_path, model, noise, QF):
    database_code = np.load(model_save_path + "database_code.npy")
    database_label = np.load(model_save_path + "database_label.npy")
    # 加载database code + label   (检索算法中计算出来的)
    database_txt_path = './data/CASIA/database.txt'
    database_img_path = np.array(open(database_txt_path).readlines())
    # 数据集
    dset = "CASIA"
    # 加载测试集
    data_path = os.path.dirname(list_path) + '/'
    # list_path = './data/CASIA/database.txt'
    list = open(list_path).readlines()
    test_loader = load_data(data_path, list, 1, 255, 224, 'test', dset)
    noise = torch.from_numpy(noise)
    noise = clamp_noise(noise, dset)
    # save_pert(test_loader, noise, img_save_path, dataset=dset, QF=QF)  # tqdm
    topk = 5
    per_codes, org_codes, org_labels = compute_result(test_loader, noise, None, model, device="cuda:0", dataset=dset, QF=QF)  # tqdm

    org_mAP, org_index, org_dist, org_crt = CalcTopMap(database_code, org_codes, database_label, org_labels, topk)
    per_mAP, per_index, per_dist, per_crt = CalcTopMap(database_code, per_codes, database_label, org_labels, topk)
    org_result_path, org_result_dist, org_result_crt = '', '', ''
    per_result_path, per_result_dist, per_result_crt = '', '', ''
    # print('org_crt:', org_crt)
    # print('per_crt:', per_crt)
    print(per_mAP, org_mAP)
    database_path = open(database_txt_path).readlines()
    for task in range(len(org_index)):
        org_index_ = org_index[task][:topk].tolist()
        per_index_ = per_index[task][:topk].tolist()
        # print(per_index_)
        # print(per_index, per_index[task], per_index_)
        for i in range(len(per_index_)):
            # print(org_index_[i], per_index_[i])
            org_res_path = '/data/UTAH_datasets/CASIA-WebFace/' + database_path[org_index_[i]].split(' ')[0]
            per_res_path = '/data/UTAH_datasets/CASIA-WebFace/' + database_path[per_index_[i]].split(' ')[0]

            # org_res_label_ = database_path[i].split(' ')[1:]
            # org_res_label = ''
            # for j in range(len(org_res_label_)):
            #     org_res_label += org_res_label_[j]
            # org_res_label = org_res_label.replace('\n', '')
            org_res_dist = org_dist[task][i]
            per_res_dist = per_dist[task][i]

            org_res_crt = org_crt[task][i]
            per_res_crt = per_crt[task][i]

            org_result_path = org_result_path + str(org_res_path) + '|'
            per_result_path = per_result_path + str(per_res_path) + '|'

            # org_result_label = org_result_label + str(org_res_label) + '|'

            org_result_dist = org_result_dist + str(org_res_dist) + '|'
            per_result_dist = per_result_dist + str(per_res_dist) + '|'

            org_result_crt = org_result_crt + str(org_res_crt) + '|'
            per_result_crt = per_result_crt + str(per_res_crt) + '|'

            # print(task, 'org_res_crt', org_res_crt)
            # print('per_res_crt', per_res_crt)
        
        org_result_path += '@'
        per_result_path += '@'
        org_result_dist += '@'
        per_result_dist += '@'
        org_result_crt += '@'
        per_result_crt += '@'

    # print(org_result_crt)
    return org_mAP, per_mAP, org_result_path, org_result_dist, org_result_crt, per_result_path, per_result_dist, per_result_crt     # (result = path, label, dist)

def save_img_multi(list_path, img_save_path, noise, QF):
    print(img_save_path, '-----------')
    # os.mkdir(img_save_path)
    # 数据集
    dset = "CASIA"
    # 加载测试集
    # data_path = '/data/UTAH_datasets/CASIA-WebFace/'
    data_path = os.path.dirname(list_path) + '/'
    # list_path = './data/CASIA/database.txt'
    print(data_path)
    list = open(list_path).readlines()
    img_name = []
    for ll in list:
        img_name_tmp = ll.split(' ')[0]
        img_name.append(img_name_tmp)
    # print('test_list =', list)
    test_loader = load_data(data_path, list, 1, 255, 224, 'test', dset)
    noise = torch.from_numpy(noise)
    noise = clamp_noise(noise, dset)
    save_pert(test_loader, noise, img_save_path, img_name, dataset=dset, QF=QF)  # tqdm

if __name__ == '__main__':
    # ######## 64bit CSQ
    save_path1 = "/data/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/"  # CSQ-ResNet34
    save_path2 = "/data/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/"  # CSQ-ResNet50
    save_path3 = "/data/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/"  # CSQ-Vgg16
    save_path4 = "/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/"  # CSQ-Vgg19

    # model
    model1 = load_model(64, 'ResNet34', save_path1 + "model.pt")
    model2 = load_model(64, 'ResNet50', save_path2 + "model.pt")
    model3 = load_model(64, 'Vgg16', save_path3 + "model.pt")
    model4 = load_model(64, 'Vgg19', save_path4 + "model.pt")

    ######### 32bit CSQ
    # save_path1 = "/data/UTAH_save/CSQ-32bit/ResNet34/CASIA/0.8719462103029044/"  # CSQ-ResNet34
    # save_path2 = "/data/UTAH_save/CSQ-32bit/ResNet50/CASIA/0.8801126067142404/"  # CSQ-ResNet50
    # save_path3 = "/data/UTAH_save/CSQ-32bit/Vgg16/CASIA/0.8087784472576253/"  # CSQ-Vgg16
    # save_path4 = "/data/UTAH_save/CSQ-32bit/Vgg19/CASIA/0.82077043324932/"  # CSQ-Vgg19
    #
    # # model
    # model1 = load_model(32, 'ResNet34', save_path1 + "model.pt")
    # model2 = load_model(32, 'ResNet50', save_path2 + "model.pt")
    # model3 = load_model(32, 'Vgg16', save_path3 + "model.pt")
    # model4 = load_model(32, 'Vgg19', save_path4 + "model.pt")

    model1 = model1.cuda()
    model2 = model2.cuda()
    model3 = model3.cuda()
    model4 = model4.cuda()

    # 加载database code + label   (检索算法中计算出来的)
    database_code1 = np.load(save_path1 + "database_code.npy")
    database_label1 = np.load(save_path1 + "database_label.npy")
    database_code2 = np.load(save_path2 + "database_code.npy")
    database_label2 = np.load(save_path2 + "database_label.npy")
    database_code3 = np.load(save_path3 + "database_code.npy")
    database_label3 = np.load(save_path3 + "database_label.npy")
    database_code4 = np.load(save_path4 + "database_code.npy")
    database_label4 = np.load(save_path4 + "database_label.npy")

    # 加载database图像path
    database_txt_path = './data/CASIA/database.txt'
    database_img_path = np.array(open(database_txt_path).readlines())

    # 数据集
    dset = "CASIA"
    # 加载测试集
    data_path = '/data/UTAH_datasets/CASIA-WebFace/'
    list_path = './data/CASIA/database.txt'
    test_loader = load_data(data_path, list_path, 1, 255, 224, 'test', dset)

    noise_path = '/data/UTAH_code/UTAP/UTAP/exp/CSQ/ResNet50/CASIA/UTAP_0/'
    noise_ = np.load(noise_path + 'best_noise.npy')

    noise = torch.from_numpy(noise_)
    noise = clamp_noise(noise, dset)

    topk = 30
    save_dir = noise_path + 'log'
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir)
    log.info("testing")

    for m in range(1):
        # if m == 0:
        #     continue
        log.info('============ mode ' + str(m) + ' =============')
        per_codes1, org_codes1, org_labels1 = compute_result(test_loader, noise, model1, device="cuda:0", dataset=dset, mode=m)  # tqdm
        org_mAP = CalcTopMap(database_code1, org_codes1, database_label1, org_labels1, topk)
        per_mAP = CalcTopMap(database_code1, per_codes1, database_label1, org_labels1, topk)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        per_codes2, org_codes2, org_labels2 = compute_result(test_loader, noise, model2, device="cuda:0", dataset=dset, mode=m)  # tqdm
        org_mAP = CalcTopMap(database_code2, org_codes2, database_label2, org_labels2, topk)
        per_mAP = CalcTopMap(database_code2, per_codes2, database_label2, org_labels2, topk)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # 对于Vgg16的:
        per_codes3, org_codes3, org_labels3 = compute_result(test_loader, noise, model3, device="cuda:0", dataset=dset, mode=m)  # tqdm
        org_mAP = CalcTopMap(database_code3, org_codes3, database_label3, org_labels3, topk)
        per_mAP = CalcTopMap(database_code3, per_codes3, database_label3, org_labels3, topk)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # 对于Vgg19的:
        per_codes4, org_codes4, org_labels4 = compute_result(test_loader, noise, model4, device="cuda:0", dataset=dset, mode=m)  # tqdm
        org_mAP = CalcTopMap(database_code4, org_codes4, database_label4, org_labels4, topk)
        per_mAP = CalcTopMap(database_code4, per_codes4, database_label4, org_labels4, topk)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

