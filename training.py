# 输出训练的记录到log文件
import numpy as np
import pickle
import os

def train(model, X_train, y_train, X_val, y_val, num_epochs=20, batch_size=128,
          learning_rate=1e-2, lr_decay=0.95, print_every=1, log_path=None):
    num_train = X_train.shape[0]
    best_val_acc = 0.0
    best_params = None

    # 历史记录
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    # 打开日志文件
    log_f = open(log_path, 'a') if log_path else None

    for epoch in range(num_epochs):
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in range(0, num_train, batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            _ = model.forward(X_batch)
            loss, acc = model.compute_loss_and_accuracy(X_batch, y_batch)
            grads = model.backward(y_batch)
            model.update(grads, learning_rate)

        learning_rate *= lr_decay

        train_epoch_loss, train_epoch_acc = model.compute_loss_and_accuracy(X_train, y_train)
        val_epoch_loss, val_epoch_acc = model.compute_loss_and_accuracy(X_val, y_val)

        train_loss_history.append(train_epoch_loss)
        train_acc_history.append(train_epoch_acc)
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)

        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        msg = f"Epoch {epoch}: Train Loss={train_epoch_loss:.4f}, Acc={train_epoch_acc:.4f} | Val Loss={val_epoch_loss:.4f}, Acc={val_epoch_acc:.4f}"
        if epoch % print_every == 0:
            print(msg)
        if log_f:
            log_f.write(msg + '\n')

    summary = f"Training complete. Best Val Acc: {best_val_acc:.4f}\n"
    print(summary)
    if log_f:
        log_f.write(summary)
        log_f.close()

    model.params = best_params
    return best_val_acc
