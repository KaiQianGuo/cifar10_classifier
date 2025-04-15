import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_weights(pkl_path, save_path=None, num_neurons=16):
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)

    if 'W1' not in model:
        raise ValueError("Model does not contain 'W1' weights.")

    W1 = model['W1']  # shape: (3072, hidden_size)
    hidden_size = W1.shape[1]

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i in range(num_neurons):
        ax = axes[i // 4, i % 4]
        weights = W1[:, i]
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        try:
            image = weights.reshape(3, 32, 32).transpose(1, 2, 0)
            ax.imshow(image)
        except Exception as e:
            print(f"Neuron {i} could not be visualized. Error: {e}")
            ax.imshow(np.zeros((32, 32, 3)))  # empty image
        ax.axis('off')
    plt.suptitle("First Layer Weights Visualization")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize first layer weights of trained model.")
    parser.add_argument('--model', type=str, required=True, help="Path to the model .pkl file")
    parser.add_argument('--output', type=str, default=None, help="Optional path to save output image")
    parser.add_argument('--num', type=int, default=16, help="Number of neurons to visualize (default: 16)")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file {args.model} not found.")

    visualize_weights(args.model, args.output, args.num)
