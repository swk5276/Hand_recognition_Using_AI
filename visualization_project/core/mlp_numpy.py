import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from pathlib import Path
import csv

# -----------------------------
# 하이퍼파라미터
# -----------------------------
INPUT_SIZE = 4096              # 64x64
HIDDEN_LAYERS = [256, 196, 128]
OUTPUT_SIZE = 111
LEARNING_RATE = 1e-4
EPOCHS = 100
BATCH_SIZE = 64
SEED = 42
np.random.seed(SEED)

# -----------------------------
# 데이터 로드
# -----------------------------
def load_data(sub_dir, file_name, output_size=OUTPUT_SIZE):
    BASE_DIR = Path(__file__).resolve().parent.parent
    file_path = BASE_DIR / 'DATA' / sub_dir / file_name
    print(f"[MLP] Loading file from: {file_path}")

    X_list, y_list = [], []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row: continue
            label = int(row[0].strip()) - 1
            pixels = [float(v) / 255.0 for v in row[1:]]
            X_list.append(pixels)
            one_hot = np.zeros(output_size, dtype=np.float32)
            one_hot[label] = 1.0
            y_list.append(one_hot)
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

def iterate_minibatches(X, Y, batch_size=64, shuffle=True):
    idx = np.arange(len(X))
    if shuffle: np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch = idx[start:end]
        if len(batch) == 0: continue
        yield X[batch], Y[batch]

# -----------------------------
# Activation / Loss / Metrics
# -----------------------------
def relu(x): return np.maximum(0, x)
def relu_derivative(x): return np.where(x > 0, 1, 0)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    if y_true.ndim > 1: y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)

# -----------------------------
# MLP 모델
# -----------------------------
class NeuralNetwork:
    def __init__(self, input_size=INPUT_SIZE, hidden_layers=HIDDEN_LAYERS,
                 output_size=OUTPUT_SIZE, learning_rate=LEARNING_RATE):
        self.learning_rate = learning_rate
        self.layer_sizes = [input_size] + list(hidden_layers) + [output_size]
        self.weights, self.biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w.astype(np.float32))
            self.biases.append(b.astype(np.float32))

    def forward(self, x, training=True):
        self.activations = [x]
        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            a = relu(z)
            self.activations.append(a)
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        a = softmax(z)
        self.activations.append(a)
        return a

    def backward(self, y_true):
        deltas = [self.activations[-1] - y_true]
        for i in reversed(range(len(self.weights) - 1)):
            delta = deltas[-1] @ self.weights[i+1].T * relu_derivative(self.activations[i+1])
            deltas.append(delta)
        deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.activations[i].T @ deltas[i]
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, Y, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            total_loss, correct = 0, 0
            for xb, yb in iterate_minibatches(X, Y, batch_size):
                out = self.forward(xb)
                loss = cross_entropy(yb, out)
                total_loss += loss * len(xb)
                pred, true = np.argmax(out,1), np.argmax(yb,1)
                correct += np.sum(pred==true)
                self.backward(yb)
            avg_loss = total_loss / len(X)
            acc = correct / len(X)
            print(f"[MLP] Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}, Acc={acc*100:.2f}%")

    def evaluate(self, X, Y):
        out = self.forward(X, training=False)
        loss = cross_entropy(Y, out)
        acc = accuracy(Y, out)
        print(f"[MLP] Test Loss={loss:.4f}, Acc={acc*100:.2f}%")
        return loss, acc

    def save(self, filepath="saved_mlp.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({"weights": self.weights, "biases": self.biases}, f)
        print(f"✅ MLP saved to {filepath}")

    def load(self, filepath="saved_mlp.pkl"):
        with open(filepath, "rb") as f:
            p = pickle.load(f)
        self.weights, self.biases = p["weights"], p["biases"]
        print(f"✅ MLP loaded from {filepath}")
