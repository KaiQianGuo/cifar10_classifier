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
    'lr_decay_rate': 0.95,
    'lr_decay_every': 10,
    'MODEL_SAVE_PATH': '/home/zhengweiguo/q2q2p/cifar10_classifier/logs/search_logs_sigmoid/best_model.pkl'
}

