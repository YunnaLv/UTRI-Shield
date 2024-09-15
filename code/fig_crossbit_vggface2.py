import matplotlib.pyplot as plt

x = [1, 2, 3, 4]

# # resnet34
fname = 'resnet34_vggface2.png'
titlename = '(ResNet34)'
y0 = [38.41, 9.68, 10.33, 12.05]
y1 = [11.25, 10.08, 8.88, 13.62]
y2 = [13.6, 2.48, 7.97, 6.68]
y3 = [89.07, 61.73, 49.88, 58.43]
# #
# # resnet50
fname = 'resnet50_vggface2.png'
titlename = '(ResNet50)'
y0 = [12.55, 25.93, 8.67, 11]
y1 = [9.05, 9.57, 8.18, 12.47]
y2 = [3.34, 13.52, 13.77, 13.08]
y3 = [71.44, 86.27, 44.72, 53.91]
# # # #
# # # # vgg16
# fname = 'vgg16_vggface2.png'
# titlename = '(Vgg16)'
# y0 = [15.72, 13.21, 30.24, 43.39]
# y1 = [7.51, 5.9, 9.2, 10.32]
# y2 = [3.21, 3.02, 82.99, 15.84]
# y3 = [47.59, 37.28, 80.89, 73.42]
# # # # #
# # # # # # vgg19
fname = 'vgg19_vggface2.png'
titlename = '(Vgg19)'
y0 = [14.36, 13.52, 23.03, 36.22]
y1 = [7.38, 6.86, 9.77, 10.28]
y2 = [3.29, 2.63, 20.63, 59.83]
y3 = [31.32, 30.39, 71.4, 78.72]

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