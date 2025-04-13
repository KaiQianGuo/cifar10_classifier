# # 数据配置
# DATA_PATH = '/home/zhengweiguo/q2q2p/cifar10_classifier/data/cifar-10-batches-py'  # CIFAR-10 数据集的路径

# # 模型超参数
# input_size = 3072  # CIFAR-10 图片是 32x32 RGB 图片，尺寸为 32*32*3 = 3072
# hidden_size = 100  # 隐藏层大小
# output_size = 10   # CIFAR-10 有 10 个类别
# activation = 'relu'  # 激活函数（'relu' 或 'sigmoid'）
# reg_lambda = 0.01  # L2 正则化强度

# # 训练超参数
# learning_rate = 0.01  # 学习率
# batch_size = 64  # 批次大小
# epochs = 50  # 训练轮数

# # 模型保存路径
# MODEL_SAVE_PATH = 'best_model.npz'  # 保存最优模型的文件路径

# # 学习率衰减设置（如果使用）
# lr_decay_rate = 0.96  # 学习率衰减率
# lr_decay_every = 10  # 每多少轮衰减一次学习率

# config.py
cfg = {
    'DATA_PATH': '/home/zhengweiguo/q2q2p/cifar10_classifier/data/cifar-10-batches-py',
    'input_size': 3072,      # 32*32*3
    'hidden_size': 128,
    'output_size': 10,
    'activation': 'relu',
    'reg_lambda': 0.01,
    'learning_rate': 0.01,
    'batch_size': 64,
    'epochs': 50,
    'lr_decay_rate': 0.96,
    'lr_decay_every': 10,
    'MODEL_SAVE_PATH': '/home/zhengweiguo/q2q2p/cifar10_classifier/logs/search_logs_sigmoid/best_model.pkl'
}

