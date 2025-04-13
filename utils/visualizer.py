import matplotlib.pyplot as plt

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc,output_path):
    """
    绘制训练集和验证集的损失与准确率曲线
    :param train_loss: 训练集的损失列表
    :param val_loss: 验证集的损失列表
    :param train_acc: 训练集的准确率列表
    :param val_acc: 验证集的准确率列表
    """
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_loss)), train_loss, label="Train Loss", color='blue')
    plt.plot(range(len(val_loss)), val_loss, label="Validation Loss", color='red')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_acc)), train_acc, label="Train Accuracy", color='blue')
    plt.plot(range(len(val_acc)), val_acc, label="Validation Accuracy", color='red')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # 显示图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.tight_layout()
    plt.show()
