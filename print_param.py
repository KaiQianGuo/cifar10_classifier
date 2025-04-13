import pickle

best_model_path_sigmoid = "/home/zhengweiguo/q2q2p/cifar10_classifier/logs/search_logs_sigmoid/best_model.pkl"

with open(best_model_path_sigmoid, 'rb') as f:
    best_params = pickle.load(f)

print("Sigmoid最佳模型参数：")
print(f"隐藏层神经元数（hidden_dim）：{best_params['hidden_dim']}")
print(f"学习率（learning_rate）：{best_params['learning_rate']}")
print(f"L2 正则系数（reg）：{best_params['reg']}")

best_model_path_relu = "/home/zhengweiguo/q2q2p/cifar10_classifier/logs/search_logs_Relu/best_model.pkl"

with open(best_model_path_relu, 'rb') as f:
    best_params = pickle.load(f)

print("Relu最佳模型参数：")
print(f"隐藏层神经元数（hidden_dim）：{best_params['hidden_dim']}")
print(f"学习率（learning_rate）：{best_params['learning_rate']}")
print(f"L2 正则系数（reg）：{best_params['reg']}")