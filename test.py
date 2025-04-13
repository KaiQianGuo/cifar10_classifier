import numpy as np
from model.three_layer_net import ThreeLayerNet
from utils.data_loader import load_cifar10
from config import cfg
# import config
import pickle

def load_model(model_path):
    with open(model_path, 'rb') as f:
        params = pickle.load(f)
    return params

def test(model, X_test, y_test):
    # 在测试集上评估模型
    loss, accuracy = model.compute_loss_and_accuracy(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    # 加载测试集数据
    _, _, X_test, y_test = load_cifar10('/home/zhengweiguo/q2q2p/cifar10_classifier/data/cifar-10-batches-py')

    # 初始化模型
    model = ThreeLayerNet(input_size=cfg['input_size'], 
                          hidden_size=cfg['hidden_size'], 
                          output_size=cfg['output_size'], 
                          activation=cfg['activation'], 
                          reg_lambda=cfg['reg_lambda'])

    # 加载模型参数
    model_params = load_model('/home/zhengweiguo/q2q2p/cifar10_classifier/best_model_relu.pkl')
    model.params = model_params

    # 测试模型在测试集上的表现
    test(model, X_test, y_test)

if __name__ == "__main__":
    main()
