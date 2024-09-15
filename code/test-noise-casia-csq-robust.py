import torch
from network import Vgg, ResNet
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.noise_utils import *
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision import transforms
from DiffJPEG import DiffJPEG
import os

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAP_robust/model/torch-model'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


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
    return model


# 数据加载(预处理 + loader)
def load_data(data_path, list_path, batch_size, resize_size, crop_size, data, dset):
    dataset = ImageList(data_path, open(list_path).readlines(),
                        transform=image_transform(resize_size, crop_size, data, dset))
    ### 原来是 shuffle = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return dataloader

# 压缩
def image_jpeg(images, quality, dataset):
    jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=quality)
    return jpeg(images, dataset)

# img + noise 计算net(dataloader + noise)结果
def compute_result(dataloader, noise, net, device, dataset, mode):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        perturbated_images = clamp_img(img + noise, dataset)
        # org   不进入if else
        if mode == 1:
            perturbated_images = image_jpeg(perturbated_images, 90, dataset)
            img = image_jpeg(img, 90, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 2:
            perturbated_images = image_jpeg(perturbated_images, 80, dataset)
            img = image_jpeg(img, 80, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 3:
            perturbated_images = image_jpeg(perturbated_images, 70, dataset)
            img = image_jpeg(img, 70, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 4:
            perturbated_images = image_jpeg(perturbated_images, 60, dataset)
            img = image_jpeg(img, 60, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 5:
            perturbated_images = image_jpeg(perturbated_images, 50, dataset)
            img = image_jpeg(img, 50, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 6:
            perturbated_images = image_jpeg(perturbated_images, 40, dataset)
            img = image_jpeg(img, 40, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)
        elif mode == 7:
            perturbated_images = image_jpeg(perturbated_images, 30, dataset)
            img = image_jpeg(img, 30, dataset)
            perturbated_images = clamp_img(perturbated_images, dataset)

        bs.append((net(perturbated_images.to(device))).data.cpu())  # 扰动图像
        bs_2.append((net(img.to(device))).data.cpu())  # 原图
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)

class logger(object):
    def __init__(self, path):
        self.path = path

    def info(self, msg):
        print(msg)
        with open(os.path.join(self.path, "log.txt"), 'a') as f:
            f.write(msg + "\n")

if __name__ == '__main__':
    ######## 64bit CSQ
    save_path1 = "/data/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/"  # CSQ-ResNet34
    save_path2 = "/data/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/"  # CSQ-ResNet50
    save_path3 = "/data/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/"  # CSQ-Vgg16
    save_path4 = "/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/"  # CSQ-Vgg19

    # model
    model1 = load_model(64, 'ResNet34', save_path1 + "model.pt")
    model2 = load_model(64, 'ResNet50', save_path2 + "model.pt")
    model3 = load_model(64, 'Vgg16', save_path3 + "model.pt")
    model4 = load_model(64, 'Vgg19', save_path4 + "model.pt")

    # ######### 32bit CSQ
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

    noise_path = '/data/UTAH_code/UTAP/UTAP/exp/CSQ/ResNet50/CASIA/UTAP_3/'
    noise_ = np.load(noise_path + 'best_noise.npy')

    noise = torch.from_numpy(noise_)
    noise = clamp_noise(noise, dset)

    topk = 300
    save_dir = noise_path + 'log'
    if os.path.exists(save_dir) is not True:
        os.system("mkdir -p {}".format(save_dir))
    log = logger(path=save_dir)
    log.info("testing")

    for m in range(8):
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


