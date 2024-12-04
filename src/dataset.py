import pickle
import numpy as np
import matplotlib.pyplot as plt

# 定义加载 CIFAR-10 数据集的函数
def load_cifar10_batch(file_path):
    """
    加载 CIFAR-10 数据集的单个批次数据。
    :param file_path: CIFAR-10 数据文件路径
    :return: (data, labels) 数据和标签
    """
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']  # 图像数据
    labels = batch['labels']  # 标签
    # 数据形状为 (N, 3 * 32 * 32)，需要重塑为 (N, 3, 32, 32)
    data = data.reshape(-1, 3, 32, 32).astype(np.float32)
    return data, labels

# 定义可视化函数
def visualize_cifar10(data, labels, classes, num_images=10):
    """
    可视化 CIFAR-10 数据。
    :param data: 图像数据，形状为 (N, 3, 32, 32)
    :param labels: 图像对应的标签
    :param classes: CIFAR-10 的类别名称列表
    :param num_images: 可视化的图像数量
    """
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        img = data[i].transpose(1, 2, 0)  # 转换为 (32, 32, 3) 格式
        # 数据归一化到 [0, 1]（CIFAR-10 原始值为 0-255）
        img = img / 255.0
        plt.imshow(img)
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.show()

# CIFAR-10 的类别名称
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 加载解压后的 .pickle 文件路径
file_path = "./cifar-10-batches-py/data_batch_1"

# 加载数据
data, labels = load_cifar10_batch(file_path)

# 打印数据形状
print(f"Data shape: {data.shape}")  # 应为 (10000, 3, 32, 32)
print(f"Labels length: {len(labels)}")  # 应为 10000

# 可视化前 10 张图片
visualize_cifar10(data, labels, cifar10_classes, num_images=100)
