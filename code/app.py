import random
import shutil
from datetime import datetime
from flask import Flask, request, session, redirect, render_template, url_for, send_file, jsonify
import torchvision.transforms as transforms
import zipfile
from numpy import *
import cv2
import os
from concurrent.futures import ThreadPoolExecutor
from celery import Celery
import numpy as np

from UTAP_Web import generate_universal_noise, gen_data_txt
from test_Web import load_model, test_noise_single, save_img_multi, test_noise_multi

# ThreadPoolExecutor管理线程池，允许在Flask应用中并发执行任务
executor = ThreadPoolExecutor(2)
app = Flask(__name__)
app.config['SQLALCHEMY_ECHO'] = True        # 启用SQLAlchemy的查询日志，便于调试数据库操作
app.config['SECRET_KEY'] = os.urandom(24)   # 生成一个随机的密钥作为Flask应用的密钥，用于加密会话等敏感数据
# 配置Celery（用来处理异步任务）的消息代理和结果后端，这里使用redis数据库作为消息代理和结果后端的存储介质
# 配置消息代理的路径，如果是在远程服务器上，则配置远程服务器中redis的URL
app.config['CELERY_BROKER_URL'] = 'redis://localhost:5431/0'
# 要存储Celery任务的状态或运行结果时就必须要配置
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:5431/0'
# 初始化Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
# 将Flask中的配置直接传递给Celery
celery.conf.update(app.config)

user_data = "user_data/"        # 用户自己的模型、照片啥的
# wm_tensor = None

# 登录，输入长度为15的01符串，如111110000011111，输入别的会报错
@app.route('/', methods=['GET', 'POST'])
def login():  # 登录，并修改全局变量user_id,
    if request.method == 'GET':
        return render_template('login.html')
    if request.method == 'POST':
        session["user_id"] = request.form.get('user_id')
        session["login"] = 'YES'
        user_id = session.get('user_id')
        # global wm_tensor
        # wm_tensor = [int(num) for num in user_id]
        return redirect(url_for('generate_noise'))

# 检索模块


# download dataset-test
@app.route("/download_test", methods=["GET", "POST"])
def download_dataset_test():
    data_test = '/data/UTAH_code/UTAP/UTAP/static/download_train.zip'
    if os.path.isfile(data_test):
        return send_file(data_test, as_attachment=True)
    else:
        return "The downloaded file does not exist"

# download dataset-attack
@app.route("/download_dataset", methods=["GET", "POST"])
def download_dataset():
    path_ = os.path.dirname(__file__)
    save_path = session.get('save_path_multi')
    zip_path = os.path.join(path_, save_path, 'protected/')
    dataset_path = os.path.join(path_, save_path, 'protected.zip')
    zip_folder(zip_path, dataset_path)

    if os.path.isfile(dataset_path):
        return send_file(dataset_path, as_attachment=True)
    else:
        return "The downloaded file does not exist"

# download dataset-test
@app.route("/download_test_multi", methods=["GET", "POST"])
def download_dataset_test_multi():
    data_test = '/data/UTAH_code/UTAP/UTAP/static/download_test.zip'
    if os.path.isfile(data_test):
        return send_file(data_test, as_attachment=True)
    else:
        return "The downloaded file does not exist"

# 主页面，通用噪声训练
@app.route('/generate_noise', methods=['GET', 'POST'])
def generate_noise():
    user_id = session.get('user_id')
    user_path = user_data + user_id + '/'
    if not os.path.exists(user_path):
        os.mkdir(user_path)
        os.mkdir(user_path + 'retrieval')
        return render_template('generate_noise.html', user_id=user_id)
    else:
        login_case = session.get('login')
        if login_case == 'YES':
            session["login"] = 'NO'
            # if os.path.exists(user_path + 'img_head_shot.png'):
            #     os.remove(user_path + 'img_head_shot.png')
            #     os.remove(user_path + 'adv.png')
            if os.path.exists(user_path + 'retrieval'):
                shutil.rmtree(user_path + 'retrieval', ignore_errors=True)
                os.mkdir(user_path + 'retrieval')
        task_id_0 = os.listdir(user_path)
        task_id = []
        for created in task_id_0:
            if created.startswith('2'):
                task_id.append(created)
        task_id = sorted(task_id)
        task_num = list(range(0, len(task_id), 1))
        task_result = []
        task_pic = []
        detail = []
        mAP = []

        for created in task_id:
            txt_path = user_path + created + '/perturbation/mAP.txt'
            # img_path = user_path + created + '/train_img/'
            pert_path = user_path + created + '/perturbation/'
            if not os.path.exists(pert_path):
                os.mkdir(pert_path)
                # shutil.copy('/data/UTAH_code/UTAP/UTAP/static/best_noise.JPEG', pert_path)
                # shutil.copy('/data/UTAH_code/UTAP/UTAP/static/mAP.txt', pert_path)

            # detail(pert_path)
            pert_name = 'best_noise.JPEG'
            pert_show = pert_path + pert_name    # 这里可能有多张.. 可能中间有的删掉了
            pert_path = pert_show
            if os.path.exists(pert_path) == False:  # 可能中间有的删掉了
                continue
            pert_stream = return_img_stream(pert_path)
            task_pic.append(pert_stream)
            if os.path.exists(txt_path):
                loss_file = open(user_path + created + '/perturbation/mAP.txt', 'r')
                result = loss_file.readlines()[-1]
                print(result)
                if result == 'END\n':
                    task_result.append('Complete')
                else:
                    task_result.append('Training')
            else:
                task_result.append('Training')

            # 读取文件, 获取retrieval_algo model_type dataset epochs
            detail_path = user_path + created + '/perturbation/detail.txt'
            with open(detail_path, 'r') as f:
                detail_ = ''
                lines = f.readlines()
                for line in lines:
                    # print(line)
                    detail_ += line
                detail.append(detail_)
            mAP_path = user_path + created + '/perturbation/mAP.txt'
            with open(mAP_path, 'r') as f:
                lines = f.readlines()
                # 去除末尾的空白行和包含 "END" 的行
                lines = [line.strip() for line in lines if line.strip() and 'END' not in line]
                if lines:   # 获取最后一行
                    last_mAP = lines[-1][:6]
                    if last_mAP != '-':
                        last_mAP = float(last_mAP) * 100
                        last_mAP = str(last_mAP)[:5]
                else:
                    last_mAP = 'None'
                mAP.append('mAP：' + last_mAP)
        return render_template('generate_noise.html', user_id=user_id, task_num=task_num, task_pic=task_pic,
                               task_time=task_id, task_result=task_result, detail=detail, mAP=mAP)

# train noise
@app.route('/train', methods=['GET', 'POST'])
def start_train_shot():
    print("train get data from user")
    user_id = session.get('user_id')
    user_path = user_data + user_id + '/'
    created = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = user_path + created + '/'
    # print('save_path', save_path)
    os.mkdir(save_path)
    session["save_path"] = save_path
    # print(session["save_path"])

    dataset = request.files.get("file_upload")  # dataset.filename
    epoch = request.form.get('epochsInput')
    epoch = int(epoch)
    model_type = request.form.get('model')
    retrieval_algo = request.form.get('algo')
    # print(epoch, model_type)
    zip_path = os.path.join(save_path, dataset.filename)  # 保存到用户ID路径下
    print('zip_path', zip_path)
    dataset.save(zip_path)
    name, _ = os.path.splitext(zip_path)
    dir_name = name.split("/")[-1]
    is_zip = zipfile.is_zipfile(zip_path)
    if is_zip:
        images = zipfile.ZipFile(zip_path)
        for image in images.namelist():
            images.extract(image, save_path)        # 提取照片
        images.close()
    else:
        return 'This is not zip'
    print("====>start process")
    old_images_path = save_path + dir_name + "/"
    images_path = save_path + "download_train/"
    os.rename(old_images_path, images_path)
    names = os.listdir(images_path)
    for image_name in names:
        img_path = images_path + image_name
        tmp = cv2.imread(img_path)
        # tmp = cv2.resize(tmp, (160, 160))
        # cv2.imwrite(img_path, tmp)
    # print("====>end process")

    # 保存detail.txt
    if os.path.exists(save_path + 'perturbation/') is False:
        os.makedirs(save_path + 'perturbation/')
        with open(save_path + 'perturbation/detail.txt', 'a+') as f:
            f.write('检索算法：' + retrieval_algo)
            f.write('，检索模型：' + model_type)
        shutil.copy('/data/UTAH_code/UTAP/UTAP/static/best_noise.JPEG', save_path + 'perturbation/')
        shutil.copy('/data/UTAH_code/UTAP/UTAP/static/mAP.txt', save_path + 'perturbation/')

    else:
        with open(save_path+'perturbation/detail.txt', 'w') as f:
            f.write('检索算法：' + retrieval_algo)
            f.write('，检索模型：' + model_type)
        shutil.copy('/data/UTAH_code/UTAP/UTAP/static/best_noise.JPEG', save_path + 'perturbation/')
        shutil.copy('/data/UTAH_code/UTAP/UTAP/static/mAP.txt', save_path + 'perturbation/')

    executor.submit(gen_data_txt, images_path)
    # generate_universal_noise(user_path, save_path, epoch, model_type, retrieval_algo)
    executor.submit(generate_universal_noise, user_path, save_path, epoch, model_type, retrieval_algo)
    return redirect(url_for('generate_noise'))

# one pic
@app.route("/retrieval_single", methods=["GET", "POST"])
def retrieval_single():
    user_id = session.get('user_id')
    if session.get('save_path') == None:
        one_org_img = user_data + user_id + '/one_org_img.png'
        adv_img_nojpeg_path = user_data + user_id + '/one_adv_img_nojpeg.png'
        adv_img_jpeg_path = user_data + user_id + '/one_adv_img_jpeg.jpg'
    else:
        save_path = session.get('save_path')
        one_org_img = save_path + 'one_org_img.png'
        adv_img_nojpeg_path = save_path + 'one_adv_img_nojpeg.png'
        adv_img_jpeg_path = save_path + 'one_adv_img_jpeg.jpg'

    dl_one_img_nojpeg, dl_one_img_jpeg = 'Y', 'Y'
    if not os.path.exists(one_org_img):
        one_org_img = '/static/img_head_shot_1.png'
        # print(img_head_shot)
    if not os.path.exists(adv_img_nojpeg_path):
        adv_img_nojpeg_path = '/static/adv_1.png'
        dl_one_img_nojpeg = 'N'
    if not os.path.exists(adv_img_jpeg_path):
        adv_img_jpeg_path = '/static/adv_1.png'
        dl_one_img_jpeg = 'N'

    one_org_img_pic = return_img_stream(one_org_img)
    # print(img_head_shot_pic)
    adv_img_nojpeg = return_img_stream(adv_img_nojpeg_path)
    adv_img_jpeg = return_img_stream(adv_img_jpeg_path)

    if dl_one_img_nojpeg == 'N':
        return render_template('retrieval_single.html', user_id=user_id,
                               one_org_img=one_org_img_pic,
                               adv_img_nojpeg=adv_img_nojpeg, adv_img_jpeg=adv_img_jpeg,
                               download_one_img_nojpeg=dl_one_img_nojpeg, download_one_img_jpeg=dl_one_img_jpeg,)
    else:
        # data_path = '/data/UTAH_datasets/CASIA-WebFace/'
        org_mAP_nojpeg = session.get('org_mAP_nojpeg')
        adv_mAP_nojpeg = session.get('adv_mAP_nojpeg')
        result_path_nojpeg = session.get('result_path_nojpeg').split('|')[:-1]
        result_label_nojpeg = session.get('result_label_nojpeg').split('|')[:-1]
        result_dist_nojpeg = session.get('result_dist_nojpeg').split('|')[:-1]

        org_mAP_jpeg = session.get('org_mAP_jpeg')
        adv_mAP_jpeg = session.get('adv_mAP_jpeg')
        result_path_jpeg = session.get('result_path_jpeg').split('|')[:-1]
        result_label_jpeg = session.get('result_label_jpeg').split('|')[:-1]
        result_dist_jpeg = session.get('result_dist_jpeg').split('|')[:-1]

        adv_img_nojpeg = return_img_stream(adv_img_nojpeg_path)
        adv_img_jpeg = return_img_stream(adv_img_jpeg_path)

        # print(result_path_jpeg)
        imgs_nojpeg = return_imgs_stream(result_path_nojpeg)
        imgs_jpeg = return_imgs_stream(result_path_jpeg)

        return render_template('retrieval_single.html', user_id=user_id,
                               one_org_img=one_org_img_pic,
                               # adv_img_shot=adv_img_shot_pic,
                               download_one_img_nojpeg=dl_one_img_nojpeg, download_one_img_jpeg=dl_one_img_jpeg,
                               org_mAP_nojpeg=org_mAP_nojpeg, adv_mAP_nojpeg=adv_mAP_nojpeg,
                               org_mAP_jpeg=org_mAP_jpeg, adv_mAP_jpeg=adv_mAP_jpeg,
                               label_nojpeg=result_label_nojpeg, label_jpeg=result_label_jpeg,
                               dist_nojpeg=result_dist_nojpeg, dist_jpeg=result_dist_jpeg,
                               adv_img_nojpeg=adv_img_nojpeg, adv_img_jpeg=adv_img_jpeg,
                               imgs_nojpeg=imgs_nojpeg, imgs_jpeg=imgs_jpeg)

def get_model(model_system):
    if model_system == '':    # 默认用通用的CSQ-ResNet50
        sys_algo = 'CSQ'
        sys_model = 'ResNet50'
        sys_hash_bit = '64'
    else:
        sys_algo = model_system.split('-')[0]
        sys_model = model_system.split('-')[1]
        sys_hash_bit = model_system.split('-')[2]

    if sys_algo == 'CSQ':
        if sys_model == 'ResNet34':
            if sys_hash_bit == '64':
                model_save_path = '/data/UTAH_save/CSQ/ResNet34/CASIA/0.8795945076653223/'
                model = load_model(64, 'ResNet34', model_save_path + "model.pt")
            elif sys_hash_bit == '32':
                model_save_path = '/data/UTAH_save/CSQ-32bit/ResNet34/CASIA/0.8719462103029044/'
                model = load_model(32, 'ResNet34', model_save_path + "model.pt")
        elif sys_model == 'ResNet50':
            if sys_hash_bit == '64':
                model_save_path = '/data/UTAH_save/CSQ/ResNet50/CASIA/0.8828318460072873/'
                model = load_model(64, 'ResNet50', model_save_path + "model.pt")
            elif sys_hash_bit == '32':
                model_save_path = '/data/UTAH_save/CSQ-32bit/ResNet50/CASIA/0.8801126067142404/'
                model = load_model(32, 'ResNet50', model_save_path + "model.pt")
        elif sys_model == 'Vgg16':
            if sys_hash_bit == '64':
                model_save_path = '/data/UTAH_save/CSQ/Vgg16/CASIA/0.8504923177106464/'
                model = load_model(64, 'Vgg16', model_save_path + "model.pt")
            elif sys_hash_bit == '32':
                model_save_path = '/data/UTAH_save/CSQ-32bit/Vgg16/CASIA/0.8087784472576253/'
                model = load_model(32, 'Vgg16', model_save_path + "model.pt")
        elif sys_model == 'Vgg19':
            if sys_hash_bit == '64':
                model_save_path = '/data/UTAH_save/CSQ/Vgg19/CASIA/0.8237669211806679/'
                model = load_model(64, 'Vgg19', model_save_path + "model.pt")
            elif sys_hash_bit == '32':
                model_save_path = '/data/UTAH_save/CSQ-32bit/Vgg19/CASIA/0.82077043324932/'
                model = load_model(32, 'Vgg19', model_save_path + "model.pt")
    elif sys_algo == 'HashNet':
        if sys_model == 'ResNet50':
            model_save_path = '/data/UTAH_save/HashNet/ResNet50/CASIA/0.6270295050518304/'
            model = executor.submit(load_model, 64, 'ResNet50', model_save_path + "model.pt")
            model = load_model(64, 'ResNet50', model_save_path + "model.pt")
        elif sys_model == 'Vgg16':
            model_save_path = '/data/UTAH_save/HashNet/Vgg16/CASIA/0.5418530565337353/'
            # model = executor.submit(load_model, 64, 'Vgg16', model_save_path + "model.pt")
            model = load_model(64, 'Vgg16', model_save_path + "model.pt")
    return model_save_path, model


@app.route("/img_upload", methods=["GET", "POST"])
def one_img_upload():
    # 获取图像
    user_id = session.get('user_id')
    created = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = user_data + user_id + '/retrieval/' + created + '-single/'
    session['save_path'] = save_path
    os.mkdir(save_path)
    # print('sp', save_path)
    org_img = request.files.get("img_upload")  # dataset.filename
    print(org_img)
    label = org_img.filename.split('.')[0]
    org_img_path = os.path.join(save_path, 'one_org_img.png')  # 保存到用户ID路径下
    org_img.save(org_img_path)
    adv_img_path_nojpeg = os.path.join(save_path, 'one_adv_img_nojpeg.png')
    adv_img_path_jpeg = os.path.join(save_path, 'one_adv_img_jpeg.jpg')

    # 获取噪声 .npy
    noise_type = request.form.get('noise_type')
    if noise_type == 'user_upload':
        uni_noise_path = request.files.get("user_upload_user")
        if uni_noise_path:
            print('testing')
            pass
        else:
            uni_noise_path = './static/noise/CSQ-ResNet50.npy'
        uni_noise = np.load(uni_noise_path)
    elif noise_type == 'prvd_noise':
        prvd = request.form.get('prvd_noise')
        if prvd == '':
            uni_noise_path = './static/noise/CSQ-ResNet50.npy'
        else:
            uni_noise_path = './static/noise/' + prvd + '.npy'
        uni_noise = np.load(uni_noise_path)
        # noise_hash_bit = int(uni_noise.split('-')[2])
    else:   # 默认用通用的CSQ-ResNet50
        uni_noise_path = './static/noise/CSQ-ResNet50.npy'
        uni_noise = np.load(uni_noise_path)
    print(uni_noise_path)

    # 获取模型
    model_system = request.form.get('system')
    if model_system == '':
        model_system = 'CSQ-ResNet50-64'
    model_save_path, model = get_model(model_system)
    QF = request.form.get('QF')
    if QF == '':
        QF = '50'
    QF = int(QF)
    org_mAP_nojpeg, adv_mAP_nojpeg, result_path_nojpeg, result_label_nojpeg, result_dist_nojpeg = test_noise_single(model_save_path, org_img_path,
                                                                        adv_img_path_nojpeg, label,
                                                                        model, uni_noise, QF=None)
    org_mAP_jpeg, adv_mAP_jpeg, result_path_jpeg, result_label_jpeg, result_dist_jpeg = test_noise_single(model_save_path, org_img_path,
                                                                  adv_img_path_jpeg, label, model, uni_noise, QF=QF)

    session['org_mAP_nojpeg'] = str(float(org_mAP_nojpeg)*100)[:5] + '%'
    session['adv_mAP_nojpeg'] = str(float(adv_mAP_nojpeg)*100)[:5] + '%'
    session['org_mAP_jpeg'] = str(float(org_mAP_jpeg)*100)[:5] + '%'
    session['adv_mAP_jpeg'] = str(float(adv_mAP_jpeg)*100)[:5] + '%'

    session['result_path_nojpeg'] = result_path_nojpeg
    session['result_label_nojpeg'] = result_label_nojpeg
    session['result_dist_nojpeg'] = result_dist_nojpeg
    session['result_path_jpeg'] = result_path_jpeg
    session['result_label_jpeg'] = result_label_jpeg
    session['result_dist_jpeg'] = result_dist_jpeg
    return redirect(url_for('retrieval_single'))


@app.route("/download_one_img_nojpeg", methods=["GET", "POST"])
def download_one_img_nojpeg():
    save_path = session.get('save_path')
    download_one_img_path = save_path + 'one_adv_img_nojpeg.png'
    if os.path.isfile(download_one_img_path):
        return send_file(download_one_img_path, as_attachment=True)
    else:
        return "The downloaded file does not exist"

@app.route("/download_one_img_jpeg", methods=["GET", "POST"])
def download_one_img_jpeg():
    save_path = session.get('save_path')
    download_one_img_path = save_path + 'one_adv_img_jpeg.jpg'
    if os.path.isfile(download_one_img_path):
        return send_file(download_one_img_path, as_attachment=True)
    else:
        return "The downloaded file does not exist"

@app.route("/retrieval_multi", methods=["GET", "POST"])
def retrieval_multi():
    user_id = session.get('user_id')
    save_path = session.get('save_path_multi')
    print('save_p', save_path)
    if save_path == None or os.path.exists(save_path) == False:
        noise_ok = 'N'
    else:   # 点两下多图检索就会出现noise_ok=Y 因为第一下创建了文件
        if os.path.exists(save_path + 'protected'):
            noise_ok = 'Y'
        else:
            noise_ok = 'N'

    retrieval_path = save_path
    if retrieval_path == None:
        retrieval_ok = 'N'
    else:
        record_path = save_path + 'mAP.txt'
        retrieval_ok = 'Y' if os.path.exists(record_path) else 'N'

    if retrieval_ok == 'Y':
        org_result_path_white = session.get('org_result_path_white').split('@')[:-1]
        org_result_crt_white = session.get('org_result_crt_white').split('@')[:-1]
        per_result_path_white = session.get('per_result_path_white').split('@')[:-1]
        per_result_crt_white = session.get('per_result_crt_white').split('@')[:-1]

        org_result_path_black = session.get('org_result_path_black').split('@')[:-1]
        org_result_crt_black = session.get('org_result_crt_black').split('@')[:-1]
        per_result_path_black = session.get('per_result_path_black').split('@')[:-1]
        per_result_crt_black = session.get('per_result_crt_black').split('@')[:-1]

        task = len(org_result_path_white)
        topk = len(org_result_path_black[0].split('|')[:-1])
        session['task'] = task
        session['topk'] = topk

        org_white_imgs, per_white_imgs, org_black_imgs, per_black_imgs = [], [], [], []
        org_white_crt, per_white_crt, org_black_crt, per_black_crt = [], [], [], []

        txt_path = save_path + 'download_test/test.txt'
        org_img, per_img = [], []
        org_img_stream, per_img_stream = [], []
        with open(txt_path, 'r') as file:
            for line in file:
                pth = line.split(' ')[0]
                org_img.append(save_path + 'download_test/' + pth)
                per_img.append(save_path + 'protected/' + pth)

        for i in range(task):
            org_path_white = org_result_path_white[i].split('|')[:-1]
            org_crt_white = org_result_crt_white[i].split('|')[:-1]
            per_path_white = per_result_path_white[i].split('|')[:-1]
            per_crt_white = per_result_crt_white[i].split('|')[:-1]

            org_path_black = org_result_path_black[i].split('|')[:-1]
            org_crt_black = org_result_crt_black[i].split('|')[:-1]
            per_path_black = per_result_path_black[i].split('|')[:-1]
            per_crt_black = per_result_crt_black[i].split('|')[:-1]

            # print(per_path_white, '\n\n', per_path_black)
            org_white_imgs.append(return_imgs_stream(org_path_white))
            per_white_imgs.append(return_imgs_stream(per_path_white))
            org_black_imgs.append(return_imgs_stream(org_path_black))
            per_black_imgs.append(return_imgs_stream(per_path_black))

            org_white_crt.append(org_crt_white)
            per_white_crt.append(per_crt_white)
            org_black_crt.append(org_crt_black)
            per_black_crt.append(per_crt_black)

            org_img_stream.append(return_img_stream(org_img[i]))
            per_img_stream.append(return_img_stream(per_img[i]))

        json_data = get_json_data(record_path)
        # json_data = get_json_data(decode_path)
        # pic_show_path = [retrieval_path + 'JPEGWhite.png', retrieval_path + 'JPEGBlack.png']
        # pic_show = []
        # for p_path in pic_show_path:
        #     img_stream = return_img_stream(p_path)
        #     pic_show.append(img_stream)

        print(org_white_crt[0][0])
        return render_template('retrieval_multi.html', user_id=user_id, retrieval_ok=retrieval_ok, noise_ok=noise_ok,
                                json_data_1=json_data[0:3], json_data_2=json_data[3:6],
                                org_white_imgs=org_white_imgs, per_white_imgs=per_white_imgs,
                                org_black_imgs=org_black_imgs, per_black_imgs=per_black_imgs,
                                org_white_crt=org_white_crt, per_white_crt=per_white_crt,
                                org_black_crt=org_black_crt, per_black_crt=per_black_crt,
                                org_img_stream=org_img_stream, per_img_stream=per_img_stream

        )
    return render_template('retrieval_multi.html', user_id=user_id, retrieval_ok=retrieval_ok, noise_ok=noise_ok)

@app.route("/noise", methods=["GET", "POST"])
def noise():
    # 获取图像
    user_id = session.get('user_id')
    created = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = user_data + user_id + '/retrieval/' + created + '-multi/'
    os.mkdir(save_path)
    session['save_path_multi'] = save_path

    # os.mkdir(save_path)
    print('sp', save_path)

    # 获取噪声 .npy
    noise_type = request.form.get('noise_type_multi')
    if noise_type == 'user_upload_multi':
        uni_noise_path = request.files.get("user_upload_multi")
        if uni_noise_path:
            pass
        else:
            uni_noise_path = './static/noise/CSQ-ResNet50.npy'
        uni_noise = np.load(uni_noise_path)
    elif noise_type == 'prvd_noise_multi':
        prvd = request.form.get('prvd_noise_multi')
        if prvd == '':
            uni_noise_path = './static/noise/CSQ-ResNet50.npy'
        else:
            uni_noise_path = './static/noise/' + prvd + '.npy'
        uni_noise = np.load(uni_noise_path)
    else:  # 默认用通用的CSQ-ResNet50
        uni_noise_path = './static/noise/CSQ-ResNet50.npy'
        uni_noise = np.load(uni_noise_path)
    # print(uni_noise)
    print('uni_noise_pth', uni_noise_path)
    # 获得解压文件: 图像+txt
    dataset = request.files.get("img_zip_upload")
    zip_path = os.path.join(save_path, dataset.filename)
    print('zip_path', zip_path)
    dataset.save(zip_path)      # 保存到用户ID路径下
    name, _ = os.path.splitext(zip_path)
    dir_name = name.split("/")[-1]
    is_zip = zipfile.is_zipfile(zip_path)
    txt_path = ''
    if is_zip:
        files = zipfile.ZipFile(zip_path)
        for file in files.namelist():
            files.extract(file, save_path)  # 提取照片+txt
            if file.endswith('.txt'):
                txt_path = save_path + file
                print('txt:', txt_path)
        files.close()
    else:
        return 'This is not zip'

    protect_path = os.path.join(save_path, 'protected/')
    if os.path.exists(protect_path) == False:
        os.mkdir(protect_path)
    print('pth', protect_path)
    np.save(protect_path + 'noise.npy', uni_noise)

    executor.submit(save_img_multi, txt_path, protect_path, uni_noise, QF=None)

    return redirect(url_for('retrieval_multi'))

@app.route("/retrieval", methods=["GET", "POST"])
def retrieval():
    save_path = session.get('save_path_multi')
    model_system_white = request.form.get('system_white')
    if model_system_white == '':
        model_system_white = 'CSQ-ResNet50-64'
    model_save_path_white, model_white = get_model(model_system_white)
    model_system_black = request.form.get('system_black')
    if model_system_black == '':
        model_system_black = 'CSQ-Vgg16-64'
    model_save_path_black, model_black = get_model(model_system_black)
    uni_noise = np.load(save_path + 'protected/noise.npy')

    QF_all = [None, 90, 80, 70, 60, 50, 40, 30]
    # QF_all = [None, 50, 30]
    # list_path
    list_path = save_path + 'download_test/test.txt'

    org_mAP_white, adv_mAP_white, org_result_path_white, org_result_crt_white, per_result_path_white, per_result_crt_white, = '', '', '', '', '', ''
    org_mAP_black, adv_mAP_black, org_result_path_black, org_result_crt_black, per_result_path_black, per_result_crt_black, = '', '', '', '', '', ''
    org_mAP_white_all, adv_mAP_white_all, org_mAP_black_all, adv_mAP_black_all = [], [], [], []

    # white
    file_path = save_path + 'mAP.txt'
    if not os.path.exists(file_path):
        os.system(r"touch {}".format(file_path))

    file_record = open(file_path, 'w')
    for QF in QF_all:
        # model_save_path, list_path, model, noise, QF
        org_mAP, adv_mAP, org_path, org_dist, org_crt, per_path, per_dist, per_crt = \
            test_noise_multi(model_save_path_white, list_path, model_white, uni_noise, QF=QF)
        if QF == None:
            org_mAP_white = str(org_mAP)
            adv_mAP_white = str(adv_mAP)
            org_result_path_white = org_path
            per_result_path_white = per_path
            org_result_crt_white = org_crt
            per_result_crt_white = per_crt
        # org_mAP_white_all.append(org_mAP)
        print(org_mAP, ',', adv_mAP, ',', QF, file=file_record)

    # black
    for QF in QF_all:
        # model_save_path, list_path, model, noise, QF
        org_mAP, adv_mAP, org_path, org_dist, org_crt, per_path, per_dist, per_crt = \
            test_noise_multi(model_save_path_black, list_path, model_black, uni_noise, QF=QF)

        if QF == None:
            org_mAP_black = str(org_mAP)
            adv_mAP_black = str(adv_mAP)
            org_result_path_black = org_path
            per_result_path_black = per_path
            org_result_crt_black = org_crt
            per_result_crt_black = per_crt
        print(org_mAP, ',', adv_mAP, ',', QF, file=file_record)

    file_record.close()

    # print(per_result_path_white, per_result_path_black)
    # print('????', str(per_result_path_white)==str(per_result_path_black))
    print('end!')
    data_path = '/data/UTAH_datasets/CASIA-WebFace/'
    session['data_path'] = data_path

    print(org_mAP_white, adv_mAP_white)

    session['org_result_path_white'] = org_result_path_white
    session['org_result_crt_white'] = org_result_crt_white
    session['per_result_path_white'] = per_result_path_white
    session['per_result_crt_white'] = per_result_crt_white

    session['org_result_path_black'] = org_result_path_black
    session['org_result_crt_black'] = org_result_crt_black
    session['per_result_path_black'] = per_result_path_black
    session['per_result_crt_black'] = per_result_crt_black

    return redirect(url_for('retrieval_multi'))

# download ckpt
@app.route("/download_ckpt/<time_path>", methods=["GET", "POST"])
def download_ckpt(time_path):
    print(time_path)
    path_ = os.path.dirname(__file__)
    print(path_)
    user_id = session.get('user_id')
    save_path = user_data + user_id + '/' + time_path + '/perturbation/'
    print(save_path)
    ckpt_path = os.path.join(path_, save_path, "best_noise.npy")
    print(ckpt_path)
    if os.path.isfile(ckpt_path):
        return send_file(ckpt_path, as_attachment=True)
    else:
        return "The downloaded file does not exist"

# download ckpt
@app.route("/delete_ckpt/<time_path>", methods=["GET", "POST"])
def delete_ckpt(time_path):
    print(time_path)
    path_ = os.path.dirname(__file__)
    user_id = session.get('user_id')
    save_path = user_data + user_id + '/' + time_path + '/'
    print(save_path)
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    return redirect(url_for('generate_noise'))


@app.route("/help", methods=["GET", "POST"])
def help_page():
    user_id = session.get('user_id')
    return render_template('help.html', user_id=user_id)

current_path = os.path.abspath(__file__)
abs_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".") + '/'

def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(abs_path + img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

def return_imgs_stream(imgs_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    imgs_stream = []
    for img_local_path in imgs_local_path:
        img_stream = ''
        with open(img_local_path, 'rb') as img_f:
            img_stream = img_f.read()
            img_stream = base64.b64encode(img_stream).decode()
            imgs_stream.append(img_stream)
    return imgs_stream


def zip_folder(folder_path, output_path):
    # 确保输出路径的文件夹存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 使用 zipfile.ZipFile 创建一个压缩文件对象
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹中的所有文件和子文件夹
        for root, _, files in os.walk(folder_path):
            for file in files:
                # 构造每个文件的完整路径
                file_path = os.path.join(root, file)
                # 将文件添加到压缩文件中的对应路径
                zipf.write(file_path, os.path.relpath(file_path, folder_path))

def get_json_data(path):
    record = open(path, 'r')
    lines = record.readlines()
    print('lines', lines)
    print(len(lines))
    white_mAP = lines[0:int(len(lines)/2)]
    black_mAP = lines[int(len(lines)/2):]

    all_list = [white_mAP, black_mAP]
    all_res = []
    for j in range(len(all_list)):
        ls = all_list[j]
        org_mAP = []
        adv_mAP = []
        delta_mAP = []
        for idx in range(len(ls)):
            line = ls[idx].split(' , ')
            print(float(str(float(line[0])*100)[:5]))
            org_mAP.append(float(str(float(line[0])*100)[:5]))
            adv_mAP.append(float(str(float(line[1])*100)[:5]))
            delta_mAP.append(float(str(float(line[0])*100)[:5]) - float(str(float(line[1])*100)[:5]))
            #
            # if idx % 2 == 1:
            #     line = ls[idx].split(',')[-4:]
            #     # print(line)
            #     acc.append(round(float(line[0][6:]), 4))
            #     acc_noise.append(round(float(line[1][1:]), 4))
            #     p.append(round(float(line[2][4:]), 3))
            #     p_noise.append(round(float(line[3][1:-2]), 3))
        data = [org_mAP, adv_mAP, delta_mAP]
        all_res.append(data)
    json_data = []
    for idx in range(len(all_list)):
        tmp_1 = {"name": "org_mAP", "data": all_res[idx][0]}
        tmp_2 = {"name": "adv_mAP", "data": all_res[idx][1]}
        tmp_3 = {"name": "delta_mAP", "data": all_res[idx][2]}
        json_data.append(tmp_1)
        json_data.append(tmp_2)
        json_data.append(tmp_3)
    print('0:3', json_data[0:3])
    print('3:6', json_data[3:6])
    return json_data

if __name__ == '__main__':
    print('start')
    app.run(debug=True, host="0.0.0.0", port=5431)

    # http://172.28.6.71:5432/
    # http://152.136.33.166:8080/