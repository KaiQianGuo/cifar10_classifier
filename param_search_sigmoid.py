# 输出结果都文件

import numpy as np
import pickle
import os
from itertools import product
from utils.data_loader import load_cifar10
from model.three_layer_net import ThreeLayerNet
from training import train

# 超参数网格
hidden_dims = [64, 128, 256]
learning_rates = [1e-1, 1e-2, 1e-3]
regularizations = [0.0, 1e-3, 1e-2]

# 加载数据
data_dir = "/home/zhengweiguo/q2q2p/cifar10_classifier/data/cifar-10-batches-py"
X_train, y_train, X_test, y_test = load_cifar10(data_dir)

# 训练/验证划分
num_train = int(0.8 * X_train.shape[0])
X_val = X_train[num_train:]
y_val = y_train[num_train:]
X_train = X_train[:num_train]
y_train = y_train[:num_train]

input_size = 32 * 32 * 3
output_size = 10

# 创建输出目录
output_dir = "/home/zhengweiguo/q2q2p/cifar10_classifier/logs/search_logs_sigmoid"
os.makedirs(output_dir, exist_ok=True)

best_val_acc = 0.0
best_params = None

# 超参搜索,此处激活函数位sigmoid
for idx, (hidden_dim, lr, reg) in enumerate(product(hidden_dims, learning_rates, regularizations)):
    config_name = f"hidden{hidden_dim}_lr{lr}_reg{reg}"
    print(f"Training with {config_name}")

    net = ThreeLayerNet(input_size, hidden_dim, output_size, activation='sigmoid', reg_lambda=reg)

    log_file = os.path.join(output_dir, f"{config_name}.log")

    val_acc = train(
        net, X_train, y_train, X_val, y_val,
        num_epochs=100, learning_rate=lr, lr_decay=0.95,
        print_every=5,
        log_path=log_file
    )

    with open(log_file, 'a') as f:
        f.write(f"Final validation accuracy: {val_acc:.4f}\n")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_params = {
            'hidden_dim': hidden_dim,
            'learning_rate': lr,
            'reg': reg,
            'model_params': {k: v.copy() for k, v in net.params.items()}
        }

# 保存最优模型
if best_params:
    with open(os.path.join(output_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_params, f)
    print(f"Best model saved with Val Acc: {best_val_acc:.4f}")
else:
    print("No valid model found.")
