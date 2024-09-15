import os
import torch.optim as optim
import time
from scipy.linalg import hadamard
import random
from network import *
from utils.tools import *

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['TORCH_HOME'] = './model/torch-model'


# ResNet34 bs = 32, epoch = 60
# ResNet50 bs = 8, epoch = 60
# ResNet18 bs = 32, epoch = 80
# Vgg11 bs = 32, epoch = 80
# Vgg16 bs = 16, epoch = 80
# AlexNet bs = 32, epoch = 130
def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "net": ResNet,
        "specific_type": "ResNet34",
        "dataset": "CASIA",
        "epoch": 80,
        "test_map": 5,
        "device": torch.device("cuda:0"),
        "bit": 64,
        "save_path": "save/CSQ",
    }
    config = config_dataset(config)
    return config


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        Q_loss = (u.abs() - 1).pow(2).mean()
        return center_loss + config["lambda"] * Q_loss

    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)  # 2bit * bit
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()
        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    break
        return hash_targets

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1  # -1 1
        return hash_center


def train_val(config):
    bit = config["bit"]
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](bit, config['specific_type']).to(device)
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    criterion = CSQLoss(config, bit)
    Best_mAP = 0
    t_loss, t_mAP = [], []
    v_loss, v_mAP = [], []

    for epoch in range(config["epoch"]):
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        # train
        net.train()
        train_loss = 0
        for image, label, _ in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)
            loss = criterion(u, label.float(), config)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)
        t_loss.append(train_loss)
        print("\b\b\b\b\b\b\b train_loss:%.3f" % (train_loss))

        # eval
        net.eval()
        val_loss = 0
        for image, label, _ in test_loader:
            image = image.to(device)
            label = label.to(device)
            u = net(image)
            loss = criterion(u, label.float(), config)
            val_loss += loss.item()

        val_loss = val_loss / len(test_loader)
        v_loss.append(val_loss)
        print("\b\b\b\b\b\b\b val_loss:%.3f" % (val_loss))

        if (epoch + 1) % config["test_map"] == 0:
            tr_code, tr_label = compute_result(train_loader, net, device=device)
            tst_code, tst_label = compute_result(test_loader, net, device=device)
            db_code, db_label = compute_result(dataset_loader, net, device=device)

            train_mAP = CalcTopMap(db_code.numpy(), tr_code.numpy(), db_label.numpy(), tr_label.numpy(), config["topK"])
            t_mAP.append(train_mAP)
            print("\ntrain_mAP =", train_mAP)
            mAP = CalcTopMap(db_code.numpy(), tst_code.numpy(), db_label.numpy(), tst_label.numpy(), config["topK"])
            v_mAP.append(mAP)
            if mAP > Best_mAP:
                Best_mAP = mAP
                if "save_path" in config:
                    save_path = config['save_path'] + "/" + config["specific_type"] + "/" + config[
                        "dataset"] + "/" + str(mAP) + "/"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    print("save in", config["save_path"])
                    np.save(os.path.join(save_path + "database_code.npy"), db_code.numpy())
                    np.save(os.path.join(save_path + "test_code.npy"), tst_code.numpy())
                    np.save(os.path.join(save_path + "train_code.npy"), tr_code.numpy())
                    np.save(os.path.join(save_path + "database_label.npy"), db_label.numpy())
                    np.save(os.path.join(save_path + "test_label.npy"), tst_label.numpy())
                    np.save(os.path.join(save_path + "train_label.npy"), tr_label.numpy())
                    torch.save(net.state_dict(), os.path.join(save_path + "model.pt"))

            print("%s epoch:%d, bit:%d, dataset:%s, mAP:%.3f, Best mAP: %.3f" %
                  (config["info"], epoch + 1, bit, config["dataset"], mAP, Best_mAP))

            t_loss_path = config['save_path'] + "/" + config["specific_type"] + "/" + config["dataset"] + "/t_loss.txt"
            t_mAP_path = config['save_path'] + "/" + config["specific_type"] + "/" + config["dataset"] + "/t_mAP.txt"
            v_loss_path = config['save_path'] + "/" + config["specific_type"] + "/" + config["dataset"] + "/v_loss.txt"
            v_mAP_path = config['save_path'] + "/" + config["specific_type"] + "/" + config["dataset"] + "/v_mAP.txt"

            with open(t_loss_path, 'w') as f:
                f.write(str(t_loss) + '\n')
            with open(t_mAP_path, 'w') as f:
                f.write(str(t_mAP) + '\n')
            with open(v_loss_path, 'w') as f:
                f.write(str(v_loss) + '\n')
            with open(v_mAP_path, 'w') as f:
                f.write(str(v_mAP) + '\n')


if __name__ == "__main__":
    config = get_config()
    train_val(config)
