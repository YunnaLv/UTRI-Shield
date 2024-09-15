import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True   # 舍弃损坏图像

# 根据不同的dataset, 设置其topk, n_class, data_path(图像存储位置前缀(如果有)), 
# 以及train test database.txt文件路径(文件中存图像路径+label)和batch_size
def config_dataset(config):
    # 实验测试了imagenet, people(Famous Iconic Women), CASIA数据集, 最终使用的是CASIA
    if config["dataset"] == "imagenet-50":
        config["topK"] = 100
        config["n_class"] = 50
        config["data_path"] = "/data2/disk1/UTAH_datasets/imagenet-50"
    elif config["dataset"] == "vggfaces2":
        config["topK"] = 300
        config["n_class"] = 28
        config["data_path"] = "/data2/disk1/UTAH_datasets/vggfaces2"
    elif config["dataset"] == "people":
        config["topK"] = 25
        config["n_class"] = 64
        config["data_path"] = ""    # 因为database.txt文件里存的是绝对路径
    elif config["dataset"] == "CASIA":
        config["topK"] = 300
        config["n_class"] = 28
        config["data_path"] = "/data2/disk1/UTAH_datasets/CASIA-WebFace"    # 因为database.txt文件里存的是绝对路径
    # 不同数据集的txt文件路径(list_path)差异在其dataset的命名 config["dataset"]
    # data: { "list_path", "batch_size" } 
    config["data"] = {
        "train": {"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
        "database": {"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
        "test": {"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config

# 这里是图像的处理, 得到Image.open打开图像, 并得到图像的标签
class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        # imgs = [(path, label)]
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform  # 图像变换

    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = Image.open(path).convert('RGB')   # Image.open打开图像 path -> jpeg
        img = self.transform(img)
        return img, label, index

    def __len__(self):
        return len(self.imgs)

# 图像变换
def image_transform(resize_size, crop_size, data, dataset):   # data(train, test, database), dataset(CASIA..)
    if data == "train":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]   # 测试集
    
    # mean和std由computeMeanAndStd.py计算而来
    if dataset == 'people':
        mean_ = [0.474, 0.433, 0.418]
        std_ = [0.309, 0.296, 0.295]
    elif dataset == 'CASIA':
        mean_ = [0.496, 0.385, 0.324]
        std_ = [0.284, 0.245, 0.236]
    elif dataset == 'imagenet':   # imagenet-50...
        mean_ = [0.485, 0.456, 0.406]
        std_ = [0.229, 0.224, 0.225]
    elif dataset == 'vggfaces2':
        mean_, std_ = [0.596, 0.456, 0.390], [0.263, 0.228, 0.219]


    return transforms.Compose([transforms.Resize(resize_size)]   # 所作变换
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=mean_, std=std_)]
                             )

# 获得dataloader(ImageList + DataLoader)
def get_data(config):
    dsets = {}          # ImageList得到dsets
    dset_loaders = {}   # DataLoader得到dset_loaders
    data_config = config["data"]    # config["data"]里有三类数据集的list_path和batch_size

    for data in ["train", "test", "database"]:
        dsets[data] = ImageList(config["data_path"], open(data_config[data]["list_path"]).readlines(),  # data_path + list_path
                                transform=image_transform(config["resize_size"], config["crop_size"], data, config['dataset']))
        print(data, len(dsets[data]))   # 数据集 + 图像张数
        
        if data == "train":  # 只有train需要打乱
            dset_loaders[data] = util_data.DataLoader(dsets[data], batch_size=data_config[data]["batch_size"],
                                                      shuffle=True, num_workers=0)
        else:
            dset_loaders[data] = util_data.DataLoader(dsets[data], batch_size=data_config[data]["batch_size"],
                                                      shuffle=False, num_workers=0)

    return dset_loaders["train"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train"]), len(dsets["test"]), len(dsets["database"])

# 传入dataloader和net, 计算出code并取出原label进行记录
def compute_result(dataloader, net, device):
    bs, clses = [], []  # output(hash code)和label
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs_ = net(img.to(device))
        bs.append(bs_.data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)   # 计算出的code和原label

# 计算汉明距离hamdis = 1 / 2 (bits - dot)
def CalcHammingDist(code1, code2):
    q = code2.shape[1]   # bits
    distH = 0.5 * (q - np.dot(code1, code2.transpose()))  # 矩阵乘法
    return distH

# mAP的计算函数
# dbCode是database的code, queryCode是test的code(预测hash code值)
# dbLabel是database的实际label, queryLabel是test的预测label(预测label)
# 检索rank topK的
def CalcTopMap(dbCode, queryCode, dbLabel, queryLabel, topk):
    num_query = queryLabel.shape[0]
    topk_map = 0
    for iter in range(num_query):
        same = (np.dot(queryLabel[iter, :], dbLabel.transpose()) > 0).astype(np.float32)   # label是不是一致(是不是检索正确)
        hammdist = CalcHammingDist(queryCode[iter, :], dbCode)   # 计算query与database所有图像的Hamming距离
        ind = np.argsort(hammdist)  # 汉明距离排序, 得到新的排序的[索引]
        same = same[ind]
        topk_same = same[0:topk]    # topK 是否检索正确
        topk_sum = np.sum(topk_same).astype(int)  # 一共几个检索正确的
        if topk_sum == 0:
            continue
        count = np.linspace(1, topk_sum, topk_sum)  # 创建数组, [1, 2, ..., topk_sum]   分子
        topk_index = np.asarray(np.where(topk_same == 1)) + 1.0  # top几的时候检索正确(0开始 所以要加1)   分母
        topk_map_ = np.mean(count / (topk_index))        # 计算对于这个query的mAP
        topk_map = topk_map + topk_map_
    topk_map = topk_map / num_query
    return topk_map

# # 计算攻击成功率
# def CalcSuccessRate(dbCode, queryCode, dbLabel, queryLabel, topk, success_mAP):
#     num_query = queryLabel.shape[0]
#     success_cnt = 0
#     for iter in range(num_query):
#         same = (np.dot(queryLabel[iter, :], dbLabel.transpose()) > 0).astype(np.float32)   # label是不是一致(是不是检索正确)
#         hammdist = CalcHammingDist(queryCode[iter, :], dbCode)   # 计算query与database所有图像的Hamming距离
#         ind = np.argsort(hammdist)  # 汉明距离排序, 得到新的排序的[索引]
#         same = same[ind]

#         topk_same = same[0:topk]    # topK 是否检索正确
#         topk_sum = np.sum(topk_same).astype(int)  # 一共几个检索正确的
#         if topk_sum == 0:
#             continue
#         count = np.linspace(1, topk_sum, topk_sum)  # 创建数组, [1, 2, ..., topk_sum]   分子

#         topk_index = np.asarray(np.where(topk_same == 1)) + 1.0  # top几的时候检索正确(0开始 所以要加1)   分母
#         topk_map = np.mean(count / (topk_index))        # 计算对于这个query的mAP
#         if topk_map <= success_mAP:
#             success_cnt += 1
#     success_rate = success_cnt / num_query
#     return success_rate