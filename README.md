# CIFAR-10 三层神经网络分类器

本项目基于 NumPy 实现了一个从零开始的三层神经网络，用于对 CIFAR-10 数据集进行图像分类。项目中包含了完整的训练、参数搜索、可视化和测试流程，支持 ReLU 和 Sigmoid 两种激活函数。

---

## 📁 项目结构

```
cifar10_classifier/
├── config.py                # 超参数配置文件
├── data/                   # 解压后的CIFAR-10 数据集（python版本）
│   ├── cifar-10-batches-py/
├── logs/                   # 参数搜索过程日志
│   ├── search_logs_Relu/
│   └── search_logs_sigmoid/
├── model/
│   └── three_layer_net.py  # 三层神经网络模型定义
├── picture/                # 可视化训练曲线
│   ├── curve_relu_1.png
│   └── curve_sigmoid_1.png
│   └── sigmoid_weights.png
│   └── relu_weights.png
├── utils/                    # 工具函数
│   ├── data_loader.py        # 数据加载器
│   └── visualizer.py         # 训练过程可视化工具
│   └── visualize_weights.py  # 模型参数可视化工具
├── best_model_relu.pkl       # ReLU 激活函数下的最佳模型
├── best_model_sigmoid.pkl    # Sigmoid 激活函数下的最佳模型
├── param_search_relu.py      # ReLU 参数搜索脚本
├── param_search_sigmoid.py   # Sigmoid 参数搜索脚本
├── print_param.py            # 打印模型最佳超参数组合
├── train_vis_relu.py         # 使用 ReLU 最佳参数训练模型并可视化
├── train_vis_sigmoid.py      # 使用 Sigmoid 最佳参数训练模型并可视化
├── training.py               # 训练逻辑主脚本（参数搜索时用）
├── test.py                   # 模型测试脚本
└── README.md                 # 项目说明文件
```

---

## 🚀 快速开始

1. **安装依赖：**

   本项目使用 Python 3，无需额外深度学习库，仅依赖基础库：

   ```bash
   pip install numpy matplotlib
   ```

2. **准备数据：**

   解压后的 CIFAR-10 数据集（`cifar-10-python.tar.gz`）保存在`data/cifar-10-batches-py` 文件夹中，运行如下脚本加载数据：

   ```python
   from utils.data_loader import load_cifar10_data
   X_train, y_train, X_test, y_test = load_cifar10_data()
   ```

3. **参数搜索：**

   ```bash
   python param_search_relu.py       # 搜索 ReLU 激活下的最佳参数组合
   python param_search_sigmoid.py    # 搜索 Sigmoid 激活下的最佳参数组合
   ```

4. **训练最佳模型并可视化：**

   ```bash
   python print_param.py # 根据输出的参数调整train_vis_relu.py和train_vis_sigmoid.py文件的超参数
   python train_vis_relu.py
   python train_vis_sigmoid.py
   ```

5. **模型权重可视化：**

   本项目支持对已训练模型的**第一层权重**进行可视化，帮助理解神经网络在输入层学习到的图像基础特征。
   可视化脚本支持读取通过 `pickle` 序列化保存的模型文件（`.pkl`），模型应包含名为 `"W1"` 的键，对应第一层权重（形状应为 `(3072, hidden_size)`）。

   通过命令行运行如下命令可生成可视化图像：

   ```bash
   python visualize_weights.py \
     --model best_model.pkl \
     --output weights.png 
   ```
6. **测试模型：**

   ```bash
   python test.py
   ```

---

## 📊 可视化效果

训练过程中，loss 和 accuracy 随 epoch 变化的图像保存在 `picture/` 目录中，分别为：
- `curve_relu_1.png`
- `curve_sigmoid_1.png`

模型参数，对应第一层权重的前16个神经元的结果可视化结果保存在 `picture/` 目录中，分别为：
- `sigmoid_weights.png`
- `relu_weights.png`

---

## 🧠 模型结构

- **输入层**：输入图像为 $32\times32\times3$ 的彩色图像，展平为 3072 维向量。
- **隐藏层**：可选神经元个数为 64、128 或 256（通过超参数搜索确定）
- **输出层**：包含 10 个神经元，使用 Softmax 函数输出每个类别的概率分布。
- 支持 ReLU 与 Sigmoid 激活函数
- 使用 Softmax 和交叉熵损失

---

## ✨ 项目亮点

- 不依赖 TensorFlow/PyTorch，完全基于 NumPy 手动实现前向传播、反向传播与梯度更新
- 支持超参数搜索
- 可视化训练过程
- 模块化代码结构，便于维护和扩展

---

## 📮 联系方式

如有问题欢迎联系 [957140619@qq.com](mailto:957140619@qq.com)
