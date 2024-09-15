import argparse
import random
from utils.noise_utils import *
from torchvision.utils import save_image
import os
from network import *
import time
from utils.tools import ImageList, CalcTopMap, image_transform
from utils.tools import compute_result as compute_result_org
from utils.votingForCenter import voting_center, voting_anchors
from utils.pic_quality import *
import torch.nn.functional as F
from DiffJPEG import DiffJPEG

os.environ['TORCH_HOME'] = '/data/UTAH_code/UTAP_robust/model/torch-model'
cpu_num = 5     # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--output_subfold', type=str, default='UTAP')
    parser.add_argument('--hash_bit', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--alpha', type=float, default=1.0)  # tanh(αx)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument('--model_root', type=str, default='/data/UTAH_save/')
    parser.add_argument('--retrieval_algo', type=str, default='CSQ')
    parser.add_argument('--model_type', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='CASIA')
    parser.add_argument('--n_class', type=int, default=28)
    parser.add_argument('--mAP', type=str, default='0.8828318460072873')
    parser.add_argument('--num_R', type=int, default=10)
    parser.add_argument('--num_M', type=int, default=10)

    parser.add_argument('--img_aug', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--topk', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_txt', type=str, default='')
    parser.add_argument('--test_txt', type=str, default='')

    parser.add_argument('--DI', type=str, default='True')
    parser.add_argument('--MI', type=str, default='True')

    parser.add_argument('--quality', type=float, default=30)  # quality越小, 压缩程度越大
    parser.add_argument('--lambda', type=float, default=0.01)

    config = parser.parse_args()
    args = vars(config)
    return args

def args_setting(args):
    path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + args[
        'mAP']
    hashcenter_path = path + '/hashcenters.npy'
    model_path = path + '/model.pt'
    args['hashcenters_path'] = hashcenter_path
    args['model_path'] = model_path

    args['output_subfold'] = args['output_subfold'] + '_'

    if args['dataset'] == 'vggfaces2':
        args['data_path'] = ''
        args['topk'] = 300
        args['train_txt'] = './data/vggfaces2/train.txt'
        args['test_txt'] = './data/vggfaces2/test.txt'

    if args['dataset'] == 'CASIA':
        args['data_path'] = '/data/UTAH_datasets/CASIA-WebFace/'
        args['topk'] = 300
        args['train_txt'] = './data/CASIA/train.txt'
        args['test_txt'] = './data/CASIA/test.txt'

    if args['dataset'] == 'vggfaces2_2':
        args['data_path'] = ''
        args['topk'] = 300
        args['train_txt'] = './data/vggfaces2_2/train.txt'
        args['test_txt'] = './data/vggfaces2_2/test.txt'

    if args['dataset'] == 'vggfaces2_3':
        args['data_path'] = ''
        args['topk'] = 300
        args['train_txt'] = './data/vggfaces2_3/train.txt'
        args['test_txt'] = './data/vggfaces2_3/test.txt'

    if args['dataset'] == 'vggfaces2_4':
        args['data_path'] = ''
        args['topk'] = 300
        args['train_txt'] = './data/vggfaces2_4/train.txt'
        args['test_txt'] = './data/vggfaces2_4/test.txt'

    return args


def load_model(args):
    if 'ResNet' in args['model_type']:
        model = ResNet(args['hash_bit'], res_model=args['model_type'])
    elif 'Vgg' in args['model_type']:
        model = Vgg(args['hash_bit'], vgg_model=args['model_type'])
    else:
        raise NotImplementedError("Only ResNet and Vgg are implemented currently.")
    model.load_state_dict(torch.load(args['model_path'], map_location=args['device']))
    model.eval()
    return model


def load_model_and_hashcenter(args):
    hashcenters = np.load(args['hashcenters_path']).astype('float32')
    model = load_model(args).to(args['device'])
    return model, hashcenters


def load_data(args, list_path, data):
    dataset = ImageList(args['data_path'], open(list_path).readlines(),
                        transform=image_transform(255, 224, data, args['dataset']))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'],
                                             shuffle=True, num_workers=args['num_workers'])
    return dataloader

# 压缩
def image_jpeg(images, args):
    quality = args['quality']
    jpeg = DiffJPEG(height=224, width=224, differentiable=True, quality=quality)
    return jpeg(images.cpu(), args['dataset']).to(args['device'])

def exp_count(args):
    count = 0
    folder_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args['dataset']
    count_path = folder_path + '/count.txt'

    if os.path.exists(folder_path) is False:
        os.makedirs(folder_path)
        with open(count_path, 'a+') as f:
            f.write(str(count))
        return count
    else:
        with open(count_path) as f:
            count = int(f.readline()) + 1
        with open(count_path, 'w') as f:
            f.write(str(count))
        return count


def compute_loss(args, batch_output, target_hash):
    products = batch_output @ target_hash.t()
    variant = torch.var(products)
    k = torch.ones_like(batch_output).sum().item() * len(target_hash)
    product_loss = products.sum() / k
    loss = - product_loss
    return loss, product_loss, variant

def compute_loss_(args, batch_output, target_hash):     # 计算哈希码相似性 MSE? HammingDist?
    # print(target_hash.shape)
    products = batch_output.mul(target_hash.sign())
    k = torch.ones_like(batch_output).sum().item()
    product_loss = products.sum() / k
    loss = - product_loss
    # print(loss)
    return loss

def attack(args, images, noise, hashcenters, model, dataset):
    sub_loss = 0
    sub_prod_loss = 0
    sub_varient_loss = 0
    grads = torch.zeros_like(noise).to(args['device'])

    overall_anchor = voting_anchors(hashcenters, num_spts=args['num_R'], hash_bit=args['hash_bit'], is_father=True, min_idx=2)
    overall_anchors = overall_anchor.unsqueeze(0).repeat(len(images) * args['num_R'], 1)

    sub_anchors = torch.as_tensor(
        voting_anchors(hashcenters, args['num_R'], args['hash_bit'], is_father=False, min_idx=2))


    adv_images = clamp_img(images + noise, dataset)
    jpeg_images = adv_images.clone()
    alpha = args['alpha']

    img_size = images[0].size()
    task_images = torch.zeros(
        [args['img_aug'], len(adv_images), img_size[0], img_size[1], img_size[2]])  # img_aug, batch, 3, 224, 224
    for q in range(args['img_aug']):
        task_images[q] = adv_images
    task_images = task_images.view(-1, img_size[0], img_size[1], img_size[2])

    ### robust
    task_images_jpeg = torch.zeros(
        [args['img_aug'], len(jpeg_images), img_size[0], img_size[1], img_size[2]])  # img_aug, batch, 3, 224, 224
    for q in range(args['img_aug']):
        task_images_jpeg[q] = jpeg_images
    task_images_jpeg = task_images_jpeg.view(-1, img_size[0], img_size[1], img_size[2])

    for o in range(args['num_M']):
        select_idx = torch.as_tensor(random.sample(range(args['num_R']), args['num_R']))
        spt_anchors = sub_anchors[select_idx]
        spt_anchors_jpeg = sub_anchors[select_idx]

        task_images = task_images.to(args['device'])
        task_images_jpeg = task_images_jpeg.to(args['device'])
        spt_deltas = torch.zeros_like(task_images).to(args['device'])
        overall_anchors = overall_anchors.to(args['device'])

        spt_deltas.requires_grad = True

        pert_images = clamp_img(input_diversity(task_images.data + spt_deltas, args['DI']), dataset).to(args['device'])
        output = model.adv_forward(pert_images, alpha)
        loss, _, _ = compute_loss(args, output, spt_anchors.to(args['device']))

        ### robust
        pert_images_jpeg = clamp_img(
            input_diversity(image_jpeg(task_images_jpeg.data + spt_deltas, args), args['DI']), dataset).to(
            args['device'])
        output_jpeg = model.adv_forward(pert_images_jpeg, alpha)
        loss_jpeg, _, _ = compute_loss(args, output_jpeg, spt_anchors_jpeg.to(args['device']))

        sim = compute_loss_(args, output_jpeg, output)
        # print('sim=', sim)      # -0.9718  本身就比较相似, 所以乘个参数对数量级进行平衡
        # print('loss=', loss)    # -0.0504  与目标不相似
        total_loss = loss + loss_jpeg - args['lambda'] * sim
        total_loss.backward()

        spt_deltas.data = spt_deltas.data + 16 / 255 * spt_deltas.grad.sign()
        spt_deltas.data = clamp_noise(spt_deltas.data, dataset)
        spt_deltas.data = clamp_img(task_images.data + spt_deltas.data, dataset) - task_images.data

        new_pert_images = clamp_img(input_diversity(task_images.data + spt_deltas, args['DI']), dataset).to(
            args['device'])
        new_outputs = model.adv_forward(new_pert_images, alpha)
        new_loss, new_product_loss, new_varient_loss = compute_loss(args, new_outputs, overall_anchor.unsqueeze(0).to(args['device']))  # 这是一批图像(batch_size)的loss

        ### robust
        new_pert_images_jpeg = clamp_img(
            input_diversity(image_jpeg(task_images_jpeg.data + spt_deltas, args), args['DI']), dataset).to(
            args['device'])
        new_outputs_jpeg = model.adv_forward(new_pert_images_jpeg, alpha)
        new_loss_jpeg, new_product_loss_jpeg, new_varient_loss_jpeg = compute_loss(args, new_outputs_jpeg, overall_anchor.unsqueeze(0).to(args['device']))  # 这是一批图像(batch_size)的loss

        sim_2 = compute_loss_(args, new_outputs_jpeg, new_outputs)
        total_new_loss = new_loss + new_loss_jpeg - args['lambda'] * sim_2
        total_new_loss.backward()

        sub_loss += new_loss.data.cpu()
        sub_prod_loss += new_product_loss.data.cpu()
        sub_varient_loss += new_varient_loss.data.cpu()
        grads += spt_deltas.grad.data.sum(0)
        spt_deltas.grad.data.zero_()

    return grads, sub_loss / args['num_M'], sub_prod_loss / args['num_M'], sub_varient_loss / args['num_M']


def generate_universal_noise(args, model, hashcenters, test_loader, train_loader, count):
    noise = noise_initialization()
    noise = torch.from_numpy(noise).to(args['device'])
    noise.requires_grad = True
    tst_mAP, Best_mAP = 0, 1.0
    momentum = torch.zeros_like(noise).to(args['device'])

    for epoch in range(args['epochs']):
        batch_grads = []

        total_loss = 0
        total_prod_loss = 0
        counter = 0
        for idx, (image, label, _) in enumerate(train_loader):
            if idx % 20 == 0:
                print(idx)
            counter = idx
            image = image.to(args['device'])
            inner_grad, sub_loss, sub_prod_loss, _ = attack(args, image, noise, hashcenters, model, args['dataset'])
            inner_grad = inner_grad / (torch.mean(torch.abs(inner_grad), (0, 1, 2), keepdim=True) + 1e-12)
            batch_grads.append(inner_grad.cpu())
            total_loss += sub_loss
            total_prod_loss += sub_prod_loss
        print('Epoch %d: Avg attack loss: %f, Avg product loss: %f' % (
        epoch, total_loss / counter, total_prod_loss / counter))
        final_grad = torch.stack(batch_grads).sum(0).to(args['device'])

        # MI
        if args['MI'] == 'True':
            # print('MI True!')
            final_grad = momentum * 0.8 + final_grad / (
                        torch.mean(torch.abs(final_grad), (0, 1, 2), keepdim=True) + 1e-12)
            momentum = final_grad
        else:
            final_grad = final_grad

        noise.data = noise.data + 0.02 * final_grad.sign()
        noise.data = clamp_noise(noise.data, args['dataset'])

        tr_mAP = train_mAP(args, train_loader, noise.clone(), model, args['dataset'])
        tst_mAP, save_noise_npy = test_mAP(args, test_loader, noise.clone(), model, count, epoch, epoch, args['dataset'])
        train_pic = compute_ssim_mse_psnr(train_loader, noise.clone().detach().cpu(), model, args['dataset'])
        print("[train] ssim =", train_pic[0], ", mse =", train_pic[1], ", psnr =", train_pic[2])
        test_pic = compute_ssim_mse_psnr(test_loader, noise.clone().detach().cpu(), model, args['dataset'])
        print("[test] ssim =", test_pic[0], ", mse =", test_pic[1], ", psnr =", test_pic[2])
        draw_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' \
                    + args['dataset'] + '/' + args['output_subfold'] + str(count) + '/draw'
        if not os.path.exists(draw_path):
            os.mkdir(draw_path)
        save_mAP_quality(draw_path, tr_mAP, tst_mAP, train_pic, test_pic)
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print('[epoch]', epoch, ', [current_time]', current_time)

        if tst_mAP < Best_mAP:
            Best_mAP = tst_mAP
            save_imgs(args, noise, epoch, count, tst_mAP)
            save_noise_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args[
                'dataset'] + '/' + args['output_subfold'] + str(count) + "/best"
            np.save(save_noise_path + '_noise.npy', save_noise_npy)

    o_mAP = org_mAP(args, test_loader, model)
    record(o_mAP, tst_mAP, test_pic, draw_path)


def save_imgs(args, noise, epoch, count, mAP):
    now = "epoch_" + str(epoch)
    path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args['dataset'] + '/' + args[
        'output_subfold'] + str(
        count)
    noise_name = '/noise_' + now + "_" + str(mAP) + '.JPEG'

    save_image(
        (noise.clone().squeeze(0) + torch.abs(torch.min(noise))) / (torch.max(noise) + torch.abs(torch.min(noise))),
        path + noise_name)


def compute_result(dataloader, noise, net, device, dataset):
    bs, bs_2, clses = [], [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        img = img.to(device)
        per_images = clamp_img(img + noise, dataset)
        bs.append((net(per_images.to(device))).data.cpu())
        bs_2.append((net(img.to(device))).data.cpu())
        clses.append(cls)
    return torch.cat(bs).sign(), torch.cat(bs_2).sign(), torch.cat(clses)


def test_mAP(args, test_loader, noise, model, count, epoch, idx, dataset):
    per_codes, org_codes, org_labels = compute_result(test_loader, noise, model, device=args['device'], dataset=dataset)
    save_path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + \
                args[
                    'mAP']
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, per_codes, db_labels, org_labels, args['topk'])
    print('test_mAP =', mAP)
    exp_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args[
        'dataset'] + '/' + args['output_subfold'] + str(count) + "/" + str(epoch) + '_' + str(idx)
    np.save(exp_path + '_per_codes.npy', per_codes.numpy())
    np.save(exp_path + '_org_codes.npy', org_codes.numpy())
    np.save(exp_path + '_org_labels.npy', org_labels.numpy())
    np.save(exp_path + '_noise.npy', noise.clone().detach().cpu().numpy())
    return mAP, noise.clone().detach().cpu().numpy()


def train_mAP(args, train_loader, noise, model, dataset):
    per_codes, org_codes, org_labels = compute_result(train_loader, noise, model, device=args['device'],
                                                      dataset=dataset)
    save_path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + \
                args[
                    'mAP']
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, per_codes, db_labels, org_labels, args['topk'])
    print("train_mAP =", mAP)
    return mAP


def save_mAP_quality(draw_path, tr_mAP, tst_mAP, train_pic, test_pic):
    train_mAP_path = draw_path + '/train_mAP.txt'
    test_mAP_path = draw_path + '/test_mAP.txt'

    train_ssim_path = draw_path + '/train_ssim.txt'
    train_mse_path = draw_path + '/train_mse.txt'
    train_psnr_path = draw_path + '/train_psnr.txt'
    test_ssim_path = draw_path + '/test_ssim.txt'
    test_mse_path = draw_path + '/test_mse.txt'
    test_psnr_path = draw_path + '/test_psnr.txt'

    with open(train_mAP_path, "a") as f:
        f.write(',' + str(tr_mAP))
    with open(test_mAP_path, "a") as f:
        f.write(',' + str(tst_mAP))

    with open(train_ssim_path, "a") as f:
        f.write(',' + str(train_pic[0]))
    with open(train_mse_path, "a") as f:
        f.write(',' + str(train_pic[1]))
    with open(train_psnr_path, "a") as f:
        f.write(',' + str(train_pic[2]))
    with open(test_ssim_path, "a") as f:
        f.write(',' + str(test_pic[0]))
    with open(test_mse_path, "a") as f:
        f.write(',' + str(test_pic[1]))
    with open(test_psnr_path, "a") as f:
        f.write(',' + str(test_pic[2]))

def record(org_mAP, tst_mAP, pic_quality, path):
    with open(path + "/record.txt", "w") as f:
        mAP_record = "org_mAP = " + str(org_mAP) + " --> " + "per_mAP = " + str(tst_mAP)
        ssim_record = "ssim = " + str(pic_quality[0])
        mse_record = "mse = " + str(pic_quality[1])
        psnr_record = "psnr = " + str(pic_quality[2])
        f.write(mAP_record + '\n')
        f.write(ssim_record + '\n')
        f.write(mse_record + '\n')
        f.write(psnr_record)

def input_diversity(img, DI='True'):
    if DI == 'True':
        # print('DI True!')
        size = img.size(2)
        resize = int(size / 0.875)

        rnd = torch.randint(size, resize + 1, (1,)).item()
        rescaled = F.interpolate(img, (rnd, rnd), mode="nearest")
        h_rem = resize - rnd
        w_hem = resize - rnd
        pad_top = torch.randint(0, h_rem + 1, (1,)).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(0, w_hem + 1, (1,)).item()
        pad_right = w_hem - pad_left
        padded = F.pad(rescaled, pad=(pad_left, pad_right, pad_top, pad_bottom))
        padded = F.interpolate(padded, (size, size), mode="nearest")

        p = torch.rand(1).item()
        if p > 0.5:
            return padded
        else:
            return img
    else:
        return img


def org_mAP(args, test_loader, model):
    org_codes, org_labels = compute_result_org(test_loader, model, args['device'])
    save_path = args['model_root'] + args['retrieval_algo'] + "/" + args['model_type'] + "/" + args['dataset'] + "/" + \
                args[
                    'mAP']
    db_codes = np.load(save_path + '/database_code.npy')
    db_labels = np.load(save_path + '/database_label.npy')
    mAP = CalcTopMap(db_codes, org_codes, db_labels, org_labels, args['topk'])
    return mAP


if __name__ == '__main__':
    current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print('[current_time]', current_time)
    args = get_args()
    args = args_setting(args)
    model, hashcenters = load_model_and_hashcenter(args)
    test_loader = load_data(args, args['test_txt'], data='test')
    train_loader = load_data(args, args['train_txt'], data='train')
    count = exp_count(args)
    exp_path = './exp/' + args['retrieval_algo'] + '/' + args['model_type'] + '/' + args['dataset'] + '/' + args[
        'output_subfold'] + str(count)
    os.makedirs(exp_path, exist_ok=True)
    generate_universal_noise(args, model, hashcenters, test_loader, train_loader, count)
