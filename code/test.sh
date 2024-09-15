#!/bin/bash

#python UTAP.py --retrieval_algo 'CSQ' --model_type 'ResNet50' --dataset 'CASIA' --mAP '0.8828318460072873' --device 'cuda:2'
#python UTAP.py --retrieval_algo 'CSQ' --model_type 'ResNet34' --dataset 'CASIA' --mAP '0.8795945076653223' --device 'cuda:2'
#python UTAP.py --retrieval_algo 'CSQ' --model_type 'Vgg16' --dataset 'CASIA' --mAP '0.8504923177106464' --device 'cuda:2'
#python UTAP.py --retrieval_algo 'CSQ' --model_type 'Vgg19' --dataset 'CASIA' --mAP '0.8237669211806679' --device 'cuda:2'

#python UTAP.py --retrieval_algo 'HashNet' --model_type 'ResNet50' --dataset 'CASIA' --mAP '0.6270295050518304' --device 'cuda:2'
python UTAP.py --retrieval_algo 'HashNet' --model_type 'Vgg16' --dataset 'CASIA' --mAP '0.5418530565337353' --device 'cuda:2'

python UTAP.py --retrieval_algo 'CSQ' --model_type 'Vgg16' --dataset 'CASIA' --mAP '0.8504923177106464' --device 'cuda:2'
python UTAP.py --retrieval_algo 'CSQ' --model_type 'Vgg19' --dataset 'CASIA' --mAP '0.8237669211806679' --device 'cuda:2'

python UTAP_2.py --retrieval_algo 'CSQ' --model_type 'ResNet34' --dataset 'vggfaces2' --mAP '0.9450292005644141' --device 'cuda:2'
#python UTAP_2.py --retrieval_algo 'CSQ' --model_type 'ResNet50' --dataset 'vggfaces2' --mAP '0.9322068760005126' --device 'cuda:2'
#python UTAP_2.py --retrieval_algo 'CSQ' --model_type 'Vgg16' --dataset 'vggfaces2' --mAP '0.9258468629124492' --device 'cuda:2'
#python UTAP_2.py --retrieval_algo 'CSQ' --model_type 'Vgg19' --dataset 'vggfaces2' --mAP '0.9172481273211995' --device 'cuda:2'
