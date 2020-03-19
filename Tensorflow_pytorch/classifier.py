import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),  # 数据集加载时，默认的图片格式是 numpy，所以通过 transforms 转换成 Tensor,图像范围[0, 255] -> [0.0,1.0]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # 使用公式进行归一化channel=（channel-mean）/std，因为transforms.ToTensor()已经把数据处理成[0,1],那么(x-0.5)/0.5就是[-1.0, 1.0]
])
train_data = datasets.CIFAR10('data', train=True,
                              download=False,
                              transform=transform)

# 处理下标
num_train = len(train_data)  # 获取训练数据的长度
indices = list(range(num_train))  # 将长度形成一个下标列表

# 取数据
split = int(np.floor(0.2 * num_train))  # np.floor 返回不大于输入参数的最大整数，该语句为取训练数据的五分之一
train_idx = indices[split:]  # 取前五分之一作为训练集

# 通过下标对训练集进行采样
train_sampler = SubsetRandomSampler(train_idx)  # 无放回地按照给定的索引列表采样样本元素

# 配置加载器
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=20,
                                           sampler=train_sampler,
                                           num_workers=0
                                           )

# 设置图片对应分类
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # 分类顺序是固定好的


# 显示图片
def EachImg(img):
    img = img / 2 + 0.5  # 将图像数据转换为0.0->1.0之间，才能正常对比度显示（以前-1.0->1.0色调对比度过大）
    plt.imshow(np.transpose(img, (1, 2,
                                  0)))  # 因为在plt.imshow在现实的时候输入的是（imagesize,imagesize,channels）,而def imshow(img,text,should_save=False)中，参数img的格式为（channels,imagesize,imagesize）,这两者的格式不一致，我们需要调用一次np.transpose函数，即np.transpose(npimg,(1,2,0))，将npimg的数据格式由（channels,imagesize,imagesize）转化为（imagesize,imagesize,channels）,进行格式的转换后方可进行显示。


# 显示前20张图片和对应分类
dataiter = iter(train_loader)  # 按批次迭代开始
images, labels = dataiter.next()  # 执行一次images.shape=torch.Size([20, 3, 32, 32]),labels.shape=torch.Size([20])
images = images.numpy()  # tensor格式转换成Numpy格式
fig = plt.figure(figsize=(25, 4))  # 画布长25宽4
for idx in np.arange(20):
    ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])  # 画布分为2行10列，现在处理第idx+1个网格
    EachImg(images[idx])  # 显示第idx张图片
    ax.set_title(classes[labels[idx]])  # 分类顺序是固定好的，所以按索引可以找到对应的下标
plt.show()

# 显示图片各通道亮度图
rgb_img = images[3]  # 选出该批次第三章图片（第一批）
channels = ['red channel', 'green channel', 'blus channel']
fig = plt.figure(figsize=(36, 36))  # 画布大小为长36宽36
for idx in np.arange(rgb_img.shape[0]):  # 三个通道依次显示
    ax = fig.add_subplot(1, 3, idx + 1)  # 一行三列，当前为第idx+1个
    img = rgb_img[idx]  # 把当前通道单独取出来作为一副灰度图
    ax.imshow(img, cmap='gray')  # 显示该灰度图
    ax.set_title(channels[idx])
    width, height = img.shape  # 取每个图象的长和宽，都为32个像素点
    thresh = img.max() / 2.5  # 作为中间值，用于更改注释的颜色，使之能够偏黑时注释为白色，偏白时注释为黑色
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][
                                             y] != 0 else 0  # 如果该像素点不为0，就将该像素点返回一个新的张量tensor,将前边的输入向量四舍五入，小数个数为第二个参数的值，默认为0
            ax.annotate(str(val), xy=(y, x),  # 每个像素点写注释，表示亮度，最亮为1.0，最暗为-1.0
                        horizontalalignment='center',
                        verticalalignment='center',
                        size=8,
                        color='white' if img[x][y] < thresh else 'black')
plt.show()
