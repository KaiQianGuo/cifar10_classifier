# 功能：加载 CIFAR-10 数据
# 解压数据


import numpy as np
import pickle
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(batch_path):
    batch = unpickle(batch_path)
    data = batch[b'data']
    labels = batch[b'labels']
    data = data.reshape((len(data), 3, 32, 32)).astype("float32") / 255.0
    data = data.transpose(0, 2, 3, 1).reshape(len(data), -1)  # 展平成3072维
    return data, np.array(labels)

def load_cifar10(data_dir):
    x_train, y_train = [], []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f"data_batch_{i}")
        data, labels = load_cifar10_batch(batch_file)
        x_train.append(data)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    test_file = os.path.join(data_dir, "test_batch")
    x_test, y_test = load_cifar10_batch(test_file)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_cifar10("/home/zhengweiguo/q2q2p/cifar10_classifier/data/cifar-10-batches-py")
    print("训练集大小:", x_train.shape, y_train.shape)
    print("测试集大小:", x_test.shape, y_test.shape)


