import numpy as np
import pickle
from utils.data_loader import load_cifar10
from model.three_layer_net import ThreeLayerNet
from utils.visualizer import plot_loss_acc  # 导入可视化函数

def train(model, X_train, y_train, X_val, y_val, num_epochs=20, batch_size=128, learning_rate=1e-2, lr_decay=0.95, print_every=1):
    num_train = X_train.shape[0]
    best_val_acc = 0.0
    best_params = None

    # 用于记录每个 epoch 的损失和准确率
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        # 打乱训练数据
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        # 批量训练
        for i in range(0, num_train, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            _ = model.forward(X_batch)
            loss, acc = model.compute_loss_and_accuracy(X_batch, y_batch)
            grads = model.backward(y_batch)
            model.update(grads, learning_rate)

        # 每个 epoch 结束后衰减学习率
        learning_rate *= lr_decay

        # 计算整个训练集的损失和准确率（也可以在每个 batch 累计平均）
        train_epoch_loss, train_epoch_acc = model.compute_loss_and_accuracy(X_train, y_train)
        train_loss_history.append(train_epoch_loss)
        train_acc_history.append(train_epoch_acc)

        # 在验证集上评估
        val_epoch_loss, val_epoch_acc = model.compute_loss_and_accuracy(X_val, y_val)
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)

        # 如果验证准确率更好，则保存当前参数
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        if epoch % print_every == 0:
            print(f"Epoch {epoch}: Train Loss = {train_epoch_loss:.4f}, Train Acc = {train_epoch_acc:.4f}, Val Loss = {val_epoch_loss:.4f}, Val Acc = {val_epoch_acc:.4f}")

        # 每 10 个 epoch 绘制一次图像（不在第 0 个 epoch 绘图）
        if epoch % 10 == 0 and epoch != 0:
            plot_loss_acc(train_loss_history, val_loss_history, train_acc_history, val_acc_history,output_path='/home/zhengweiguo/q2q2p/cifar10_classifier/picture/curve_sigmoid_1.png')

    print("Training complete. Best Val Acc:", best_val_acc)
    model.params = best_params
    with open('/home/zhengweiguo/q2q2p/cifar10_classifier/best_model_sigmoid.pkl', 'wb') as f:
        pickle.dump(model.params, f)

if __name__ == '__main__':
    # 加载数据（返回训练集和测试集）
    X_train, y_train, X_test, y_test = load_cifar10("/home/zhengweiguo/q2q2p/cifar10_classifier/data/cifar-10-batches-py")
    
    input_size = 32 * 32 * 3
    hidden_size = 256
    output_size = 10

    # 手动划分验证集：取 80% 作为训练集，20% 作为验证集
    num_total = X_train.shape[0]
    num_train = int(0.8 * num_total)
    X_val = X_train[num_train:]
    y_val = y_train[num_train:]
    X_train = X_train[:num_train]
    y_train = y_train[:num_train]

    # 激活函数sigmoid，隐藏层神经元数（hidden_dim）：256，学习率（learning_rate）：0.1，L2 正则系数（reg）：0.0
    net = ThreeLayerNet(input_size, hidden_size, output_size, activation='sigmoid', reg_lambda=0.0)
    train(net, X_train, y_train, X_val, y_val, num_epochs=100, learning_rate=0.1, lr_decay=0.95)
