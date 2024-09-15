import matplotlib.pyplot as plt

x = [1, 2, 3, 4]

# # resnet34
fname = 'resnet34_casia.png'
titlename = '(ResNet34)'
y0 = [41.58, 23.58, 17.27, 18.07]
y1 = [12.01, 20.33, 15.87, 16.71]
y2 = [45.34, 2.52, 21.92, 15.2]
y3 = [84.87, 66.61, 55.88, 55.88]
# #
# resnet50
fname = 'resnet50_casia.png'
titlename = '(ResNet50)'
y0 = [21.01, 42.8, 15.48, 15.84]
y1 = [12.42, 18.12, 15.18, 16.22]
y2 = [4.28, 10.51, 17.67, 14.59]
y3 = [69.75, 81.06, 45.5, 54.94]
# # # #
# # # vgg16
fname = 'vgg16_casia.png'
titlename = '(Vgg16)'
y0 = [21.24, 22.97, 31.43, 28.87]
y1 = [8.8, 14.76, 14.15, 15.06]
y2 = [3.99, 2.25, 70.74, 32.46]
y3 = [43.99, 35.53, 79.73, 60.45]
# # # #
# # # # # # # vgg19
fname = 'vgg19_casia.png'
titlename = '(Vgg19)'
y0 = [16.32, 18.13, 20.98, 31.29]
y1 = [9.9, 15.01, 18.63, 20.3]
y2 = [4.09, 2.05, 27.39, 74.77]
y3 = [59.71, 61.15, 67.2, 78.04]

plt.figure(figsize=(4, 3))
# plt.plot(x, y0, marker='o', label='Org')
plt.plot(x, y0, marker='D', label='DHTA', color='#F3A332')#018A67
plt.plot(x, y1, marker='8', label='CWDM', color='#018A67')#1868B2
plt.plot(x, y2, marker='s', label='AdvHash', color='#1868B2')#F3A332
plt.plot(x, y3, marker='^', label='UTAP', color='#DE582B')

# plt.title('Comparison of ΔmAP ' + titlename)  # 设置标题
# plt.xlabel('Quality Factor')  # 设置x轴标签
x_labels = ['ResNet34', 'ResNet50', 'VGG16', 'VGG19*']
plt.xticks(x, x_labels)
# plt.ylabel('ΔmAP')  # 设置y轴标签
# plt.legend(loc='upper right', ncol=5, frameon=False)

plt.grid(True)
plt.tight_layout()
plt.ylim(0, 90)
plt.yticks(range(0, 91, 20))
plt.savefig('./robust_fig/' + fname)
plt.show()