import torch
from torch import nn
from network import Vgg, ResNet
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.noise_utils import *
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from DiffJPEG import DiffJPEG
import os

import torchvision.models as models

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAP_robust/model/torch-model'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

cpu_num = 1     # 这里设置成你想运行的CPU个数
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
    return model

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
def load_data(data_path, list_path, batch_size, resize_size, crop_size, data, dset):
    dataset = ImageList(data_path, open(list_path).readlines(),
                        transform=image_transform(resize_size, crop_size, data, dset))
    ### 原来是 shuffle = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

# resize
def image_resize(images, new_size):
    # new_size = 100  # ?
    transform_resize = transforms.Compose([
        transforms.Resize((new_size, new_size)),
        transforms.Resize((224, 224)),
    ])
    return transform_resize(images)

# 模糊
def image_GaussianBlur(images, ker, cir_len):
    # ker, cir_len = 3, 30     # ?  3 50/ 5 50
    # ker, cir_len = 5, 50
    transform_blur = transforms.Compose([
        transforms.GaussianBlur(ker, cir_len)
    ])
    return transform_blur(images)

# 高斯噪声
def image_noise(images, noise_v):
    # noise_v = 0.002 ** 0.5  # ?
    # noise_v = 0.005 ** 0.5
    mask = np.random.normal(0, noise_v, (3, 224, 224))
    mask = torch.from_numpy(mask).float()
    return images + mask

# 压缩
def image_jpeg(images, quality, dataset):
    # quality = 80    # ? 越大压缩程度越小
    # quality = 50
    # quality = 10
    jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=quality)
    return jpeg(images, dataset)

# img + noise 计算net(dataloader + noise)结果
def compute_result(dataloader, mask, applied_patch, net, device, dataset, mode):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        new_shape = list(mask.shape)  # mask.shape转为list形式, 比如 = [3, 200, 200]
        new_shape.insert(0, img.shape[0])  # n_img, 3, 200, 200

        perturbated_images = torch.mul(mask.type(torch.FloatTensor), applied_patch.type(torch.FloatTensor)) + \
                             torch.mul(1 - mask.expand(new_shape).type(torch.FloatTensor),
                                       img.type(torch.FloatTensor))
        # save_image(img, '1_org.png')
        save_image(un_normalize(perturbated_images[0], dataset), '2_attack.png')
        return

        # # org   不进入if else
        #
        # # 1. resize
        # if mode == 1:
        #     perturbated_images = image_resize(perturbated_images, 168)
        #     img = image_resize(img, 168)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # if mode == 2:
        #     perturbated_images = image_resize(perturbated_images, 280)
        #     img = image_resize(img, 280)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # if mode == 3:
        #     perturbated_images = image_resize(perturbated_images, 336)
        #     img = image_resize(img, 336)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # if mode == 4:
        #     perturbated_images = image_resize(perturbated_images, 392)
        #     img = image_resize(img, 392)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # # 2. 模糊
        # elif mode == 5:
        #     perturbated_images = image_GaussianBlur(perturbated_images, 3, 10)
        #     img = image_GaussianBlur(img, 3, 10)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 6:
        #     perturbated_images = image_GaussianBlur(perturbated_images, 5, 10)
        #     img = image_GaussianBlur(img, 5, 10)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 7:
        #     perturbated_images = image_GaussianBlur(perturbated_images, 7, 10)
        #     img = image_GaussianBlur(img, 7, 10)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 8:
        #     perturbated_images = image_GaussianBlur(perturbated_images, 9, 10)
        #     img = image_GaussianBlur(img, 9, 10)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # # 3. 高斯噪声
        # elif mode == 9:
        #     perturbated_images = image_noise(perturbated_images, 0.001 ** 0.5)
        #     img = image_noise(img, 0.001 ** 0.5)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 10:
        #     perturbated_images = image_noise(perturbated_images, 0.002 ** 0.5)
        #     img = image_noise(img, 0.002 ** 0.5)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 11:
        #     perturbated_images = image_noise(perturbated_images, 0.003 ** 0.5)
        #     img = image_noise(img, 0.003 ** 0.5)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 12:
        #     perturbated_images = image_noise(perturbated_images, 0.004 ** 0.5)
        #     img = image_noise(img, 0.004 ** 0.5)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        #     # 4. 压缩
        # elif mode == 13:
        #     perturbated_images = image_jpeg(perturbated_images, 90, dataset)
        #     img = image_jpeg(img, 90, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 14:
        #     perturbated_images = image_jpeg(perturbated_images, 80, dataset)
        #     img = image_jpeg(img, 80, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 15:
        #     perturbated_images = image_jpeg(perturbated_images, 70, dataset)
        #     img = image_jpeg(img, 70, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 16:
        #     perturbated_images = image_jpeg(perturbated_images, 60, dataset)
        #     img = image_jpeg(img, 60, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 17:
        #     perturbated_images = image_jpeg(perturbated_images, 50, dataset)
        #     img = image_jpeg(img, 50, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 18:
        #     perturbated_images = image_jpeg(perturbated_images, 40, dataset)
        #     img = image_jpeg(img, 40, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 19:
        #     perturbated_images = image_jpeg(perturbated_images, 30, dataset)
        #     img = image_jpeg(img, 30, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)

        # # 补一些 resize2/4, blur9, noise0.005, jpeg10
        # elif mode == 17:
        #     perturbated_images = image_resize(perturbated_images, 112)
        #     img = image_resize(img, 112)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 18:
        #     perturbated_images = image_GaussianBlur(perturbated_images, 9, 10)
        #     img = image_GaussianBlur(img, 9, 10)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 19:
        #     perturbated_images = image_noise(perturbated_images, 0.005 ** 0.5)
        #     img = image_noise(img, 0.005 ** 0.5)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # elif mode == 20:
        #     perturbated_images = image_jpeg(perturbated_images, 10, dataset)
        #     img = image_jpeg(img, 10, dataset)
        #     perturbated_images = clamp_img(perturbated_images, dataset)
        # save_image(un_normalize(img[0], dataset), './img/' + str(mode) + '_org.png')
        # save_image(un_normalize(perturbated_images[0], dataset), './img/' + str(mode) + '_attack.png')
        # return
        bs.append((net(perturbated_images.to(device))).data.cpu())  # 扰动图像
        bs_2.append((net(img.to(device))).data.cpu())  # 原图
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)

def mask_generation(patch, img_size=(3, 224, 224)):
    applied_patch = np.zeros(img_size)
    x_location = 200 - patch.shape[1]
    y_location = 200 - patch.shape[2]
    applied_patch[:, x_location: x_location + patch.shape[1], y_location: y_location + patch.shape[2]] = patch
    mask = applied_patch.copy()
    mask[mask != 0] = 1.0
    return mask, applied_patch, x_location, y_location

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

if __name__ == '__main__':
    ######## 64bit CSQ
    ######## 64bit CSQ
    save_path1 = "/data/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/"  # CSQ-ResNet34
    save_path2 = "/data/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/"  # CSQ-ResNet50
    save_path3 = "/data/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/"  # CSQ-Vgg16
    save_path4 = "/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/"  # CSQ-Vgg19

    model1 = load_model(64, 'ResNet34', save_path1 + "model.pt")
    model2 = load_model(64, 'ResNet50', save_path2 + "model.pt")
    model3 = load_model(64, 'Vgg16', save_path3 + "model.pt")
    model4 = load_model(64, 'Vgg19', save_path4 + "model.pt")
    #
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
    data_path = '/data/UTAH_code/UTAP/UTAP/save_img/'
    list_path = '/data/UTAH_code/UTAP/UTAP/save_img/1.txt'
    test_loader = load_data(data_path, list_path, 1, 255, 224, 'test', dset)

    # 加载noise
    patch_path = '/data/UTAH_code/UTAP_robust/exp/CSQ/ResNet50/CASIA_AdvHash/experiment_1-0.2/'
    patch_ = np.load(patch_path + '0_43_patch.npy')
    mask, applied_patch, x, y = mask_generation(patch_)
    applied_patch = torch.from_numpy(applied_patch)
    mask = torch.from_numpy(mask)

    topk = 300
    save_dir = patch_path + 'log'
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir)
    log.info("testing")

    # for m in range(1):
    for m in [0]:
    #     if m == 5:
    #         continue
        log.info('============ mode ' + str(m) + ' =============')
        per_codes1, org_codes1, org_labels1 = compute_result(test_loader, mask, applied_patch, model1, device="cuda:0", dataset=dset, mode=m)
        org_mAP = CalcTopMap(database_code1, org_codes1, database_label1, org_labels1, topk)
        per_mAP = CalcTopMap(database_code1, per_codes1, database_label1, org_labels1, topk)
        # print("mAP:", org_mAP, "->", per_mAP)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        per_codes2, org_codes2, org_labels2 = compute_result(test_loader, mask, applied_patch, model2, device="cuda:0", dataset=dset, mode=m)
        org_mAP = CalcTopMap(database_code2, org_codes2, database_label2, org_labels2, topk)
        per_mAP = CalcTopMap(database_code2, per_codes2, database_label2, org_labels2, topk)
        # print("mAP:", org_mAP, "->", per_mAP)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # 对于Vgg16的:
        per_codes3, org_codes3, org_labels3 = compute_result(test_loader, mask, applied_patch, model3, device="cuda:0", dataset=dset, mode=m)
        org_mAP = CalcTopMap(database_code3, org_codes3, database_label3, org_labels3, topk)
        per_mAP = CalcTopMap(database_code3, per_codes3, database_label3, org_labels3, topk)
        # print("mAP:", org_mAP, "->", per_mAP)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # 对于Vgg19的:
        per_codes4, org_codes4, org_labels4 = compute_result(test_loader, mask, applied_patch, model4, device="cuda:0", dataset=dset, mode=m)
        org_mAP = CalcTopMap(database_code4, org_codes4, database_label4, org_labels4, topk)
        per_mAP = CalcTopMap(database_code4, per_codes4, database_label4, org_labels4, topk)
        # print("mAP:", org_mAP, "->", per_mAP)
        log.info("mAP:" + str(org_mAP) + "->" + str(per_mAP))

        # # save npy文件
        # np.save('./codes/1_org_codes_resnet34.npy', org_codes1.numpy())
        # np.save('./codes/2_org_codes_resnet50.npy', org_codes2.numpy())
        # np.save('./codes/3_org_codes_vgg11.npy', org_codes3.numpy())
        # np.save('./codes/4_org_codes_vgg16.npy', org_codes4.numpy())
        #
        # np.save('./codes/5_per_codes_resnet34.npy', per_codes1.numpy())
        # np.save('./codes/6_per_codes_resnet50.npy', per_codes2.numpy())
        # np.save('./codes/7_per_codes_vgg11.npy', per_codes3.numpy())
        # np.save('./codes/8_per_codes_vgg16.npy', per_codes4.numpy())

        # # hashnet
        # np.save('./codes/1_org_codes_resnet50.npy', org_codes1.numpy())
        # np.save('./codes/2_org_codes_vgg16.npy', org_codes2.numpy())
        # np.save('./codes/3_per_codes_resnet50.npy', per_codes1.numpy())
        # np.save('./codes/4_per_codes_vgg16.npy', per_codes2.numpy())


