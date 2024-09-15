import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5, 6, 7, 8]

# # resnet34
# fname = 'resnet34-delta_v1.png'
# titlename = '(ResNet34)'
# y0 = [4.73, 3.99, 3.89, 3.85, 3.91, 3.89, 3.79, 3.77]
# y1 = [19.26, 14.32, 10.4, 7.09, 5.57, 4.48, 3.73, 3.53]
# y2 = [13.13, 13.65, 12.01, 10.15, 8.63, 7.67, 6.64, 5.66]
# y3 = [57.4, 50.73, 46.18, 42.72, 40.35, 39.15, 38.62, 35.74]
# #
# resnet50
fname = 'resnet50-delta_v1.png'
titlename = '(ResNet50)'
y0 = [85.39, 4.13, 3.81, 3.64, 3.62, 3.24, 2.99, 2.41]
y1 = [33.73, 20.59, 10.47, 6.36, 4.65, 4.05, 3.46, 3.1]
y2 = [12.91, 11.48, 9.43, 8.21, 7.19, 6.36, 5.95, 5.05]
y3 = [85.74, 83.85, 81.64, 79.79, 78.31, 77.39, 76.85, 74.81]
# # # #
# # # vgg16
# fname = 'vgg16-delta_v1.png'
# titlename = '(Vgg16)'
# y0 = [9.72, 9.06, 8.9, 8.82, 8.36, 8.3, 8.3, 8.47]
# y1 = [11.72, 10.75, 9.12, 7.47, 6.4, 5.85, 5.67, 5.16]
# y2 = [15.85, 15.86, 14.87, 13.19, 11.4, 10.61, 10.31, 9.58]
# y3 = [44.21, 39.68, 36.23, 34.02, 32.25, 31.63, 31.76, 30.31]
# # # #
# # # # # # vgg19
# fname = 'vgg19-delta_v1.png'
# titlename = '(Vgg19)'
# y0 = [20.74, 17.8, 17.47, 17.47, 17.27, 16.14, 16.69, 15.68]
# y1 = [19.2, 14.93, 11.32, 8.2, 6.48, 5.2, 4.44, 4.32]
# y2 = [15.6, 14.04, 11.54, 9.81, 8.56, 7.3, 6.68, 6.04]
# y3 = [61.11, 57.53, 54.04, 51.11, 48.46, 46.38, 45.66, 42.44]

plt.figure(figsize=(7, 4))
# plt.plot(x, y0, marker='o', label='Org')
plt.plot(x, y1, marker='D', label='DHTA', color='#F3A332')
plt.plot(x, y2, marker='8', label='CWDM', color='#018A67')
plt.plot(x, y0, marker='s', label='AdvHash', color='#1868B2')
plt.plot(x, y3, marker='^', label='UTRAP', color='#DE582B')

# plt.title('Comparison of ΔmAP ' + titlename)  # 设置标题
# plt.xlabel('Quality Factor')  # 设置x轴标签
x_labels = ['N/A', '90', '80', '70', '60', '50', '40', '30']
plt.xticks(x, x_labels)
# plt.ylabel('ΔmAP')  # 设置y轴标签
plt.legend(loc='upper right', ncol=5, frameon=False)

# plt.grid(True)
plt.tight_layout()
plt.ylim(0, 90)
plt.yticks(range(0, 91, 20))
plt.savefig('./robust_fig/title' + fname)
plt.show()