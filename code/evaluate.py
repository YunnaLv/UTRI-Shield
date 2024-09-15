import torch
from network import *
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.noise_utils import *
from tqdm import tqdm

import os

os.environ['TORCH_HOME'] = './model/torch-model'
import setGPU

def load_model(hash_bit, specific_model, model_path):
    if 'ResNet' in specific_model:
        model = ResNet(hash_bit, specific_model)  # hash_bit and specific model type
    elif 'Vgg' in specific_model:
        model = Vgg(hash_bit, specific_model)
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    return model


def load_data(data_path, list_path, batch_size, resize_size, crop_size, data, dset):
    dataset = ImageList(data_path, open(list_path).readlines(),
                        transform=image_transform(resize_size, crop_size, data, dset))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return dataloader


def compute_result(dataloader, noise, net, device, dataset):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        perturbated_images = clamp_img(img + noise, dataset)
        bs.append((net(perturbated_images.to(device))).data.cpu())
        bs_2.append((net(img.to(device))).data.cpu())
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)


if __name__ == '__main__':
    save_path1 = "/data2/disk1/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/"  # CSQ-ResNet34
    save_path2 = "/data2/disk1/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/"  # CSQ-ResNet50
    save_path3 = "/data2/disk1/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/"  # CSQ-Vgg19
    save_path4 = "/data2/disk1/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/"  # CSQ-Vgg16

    # model
    model1 = load_model(64, 'ResNet34', save_path1 + "model.pt")
    model2 = load_model(64, 'ResNet50', save_path2 + "model.pt")
    model3 = load_model(64, 'Vgg19', save_path3 + "model.pt")
    model4 = load_model(64, 'Vgg16', save_path4 + "model.pt")

    model1 = model1.cuda()
    model2 = model2.cuda()
    model3 = model3.cuda()
    model4 = model4.cuda()

    database_code1 = np.load(save_path1 + "database_code.npy")
    database_label1 = np.load(save_path1 + "database_label.npy")
    database_code2 = np.load(save_path2 + "database_code.npy")
    database_label2 = np.load(save_path2 + "database_label.npy")
    database_code3 = np.load(save_path3 + "database_code.npy")
    database_label3 = np.load(save_path3 + "database_label.npy")
    database_code4 = np.load(save_path4 + "database_code.npy")
    database_label4 = np.load(save_path4 + "database_label.npy")

    database_txt_path = './data/CASIA/database.txt'
    database_img_path = np.array(open(database_txt_path).readlines())

    dset = "CASIA"
    data_path = '/data2/disk1/UTAH_datasets/CASIA-WebFace/'
    list_path = './data/CASIA/database.txt'
    test_loader = load_data(data_path, list_path, 1, 255, 224, 'test', dset)

    model_name = 'ResNet34/CASIA/'
    noise_root = './exp/CSQ/%s' % model_name
    noise_path = 'ablation_13_maxcase__5'
    noise_ = np.load(noise_root + noise_path + '/74_74_noise.npy')
    noise = torch.from_numpy(noise_)
    noise = clamp_noise(noise, dset)

    topk = 300
    save_root = './eval_codes'
    save_code_path = save_root + '/' + model_name
    os.makedirs(save_code_path, exist_ok=True)

    # ResNet34：
    per_codes1, org_codes1, org_labels1 = compute_result(test_loader, noise, model1, device="cuda", dataset=dset)  # tqdm
    org_mAP = CalcTopMap(database_code1, org_codes1, database_label1, org_labels1, topk)
    per_mAP = CalcTopMap(database_code1, per_codes1, database_label1, org_labels1, topk)
    print("ResNet34 mAP:", org_mAP, "->", per_mAP)
    np.save(save_code_path + '1_org_codes_resnet34.npy', org_codes1.numpy())
    np.save(save_code_path + '5_per_codes_resnet34.npy', per_codes1.numpy())

    # ResNet50：
    per_codes2, org_codes2, org_labels2 = compute_result(test_loader, noise, model2, device="cuda", dataset=dset)  # tqdm
    org_mAP = CalcTopMap(database_code2, org_codes2, database_label2, org_labels2, topk)
    per_mAP = CalcTopMap(database_code2, per_codes2, database_label2, org_labels2, topk)
    print("ResNet50 mAP:", org_mAP, "->", per_mAP)
    np.save(save_code_path + '2_org_codes_resnet50.npy', org_codes2.numpy())
    np.save(save_code_path + '6_per_codes_resnet50.npy', per_codes2.numpy())

    # Vgg16:
    per_codes4, org_codes4, org_labels4 = compute_result(test_loader, noise, model4, device="cuda", dataset=dset)  # tqdm
    org_mAP = CalcTopMap(database_code4, org_codes4, database_label4, org_labels4, topk)
    per_mAP = CalcTopMap(database_code4, per_codes4, database_label4, org_labels4, topk)
    print("Vgg16 mAP:", org_mAP, "->", per_mAP)
    np.save(save_code_path + '4_org_codes_vgg16.npy', org_codes4.numpy())
    np.save(save_code_path + '8_per_codes_vgg16.npy', per_codes4.numpy())

    # Vgg19:
    per_codes3, org_codes3, org_labels3 = compute_result(test_loader, noise, model3, device="cuda", dataset=dset)  # tqdm
    org_mAP = CalcTopMap(database_code3, org_codes3, database_label3, org_labels3, topk)
    per_mAP = CalcTopMap(database_code3, per_codes3, database_label3, org_labels3, topk)
    print("Vgg19 mAP:", org_mAP, "->", per_mAP)
    np.save(save_code_path + '3_org_codes_vgg19.npy', org_codes3.numpy())
    np.save(save_code_path + '7_per_codes_vgg19.npy', per_codes3.numpy())

