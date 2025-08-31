import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os
from pathlib import Path
import pickle

# -----------------------------
# 하이퍼파라미터
# -----------------------------
img_h, img_w, in_channels = 64, 64, 1
output_size = 111
learning_rate = 1e-3
epochs = 30
batch_size = 64
seed = 42
np.random.seed(seed)

# -----------------------------
# 데이터 로더 (레이블 0-인덱스/1-인덱스 자동 인식)
# -----------------------------
def load_data(sub_dir, file_name, output_size):
    BASE_DIR = Path(__file__).resolve().parent.parent  # visualization_project 기준
    file_path = BASE_DIR / 'DATA' / sub_dir / file_name
    print(f"Loading file from: {file_path}")

    X_list, y_list = [], []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = [v.strip() for v in row if v.strip() != ""]
            if not row:
                continue
            label_raw = int(row[0])
            pixels = np.array([float(v) for v in row[1:]], dtype=np.float32) / 255.0
            # 4096 보정
            if pixels.size != img_h * img_w:
                raise ValueError(f"Expected 4096 pixels but got {pixels.size}")
            X_list.append(pixels)
            y_list.append(label_raw)

    y_arr = np.array(y_list, dtype=np.int32)
    # 0-index vs 1-index 히ュー리스틱
    y_min, y_max = y_arr.min(), y_arr.max()
    if y_min == 1 and (y_max == output_size or y_max == output_size):  # 1..111
        y_arr = y_arr - 1  # 0..110로 변환

    # 원-핫
    Y = np.zeros((len(y_arr), output_size), dtype=np.float32)
    Y[np.arange(len(y_arr)), y_arr] = 1.0

    X = np.array(X_list, dtype=np.float32)
    return X, Y

def iterate_minibatches(X, Y, batch_size=64, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch = idx[start:end]
        if len(batch) == 0:
            continue
        yield X[batch], Y[batch]

# -----------------------------
# 유틸: Softmax, Cross-Entropy, Accuracy
# -----------------------------
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def accuracy(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred)

# -----------------------------
# im2col/col2im (Conv/Pool 가속)
# -----------------------------
def get_im2col_indices(x_shape, field_h, field_w, padding, stride):
    N, C, H, W = x_shape
    out_h = (H + 2 * padding - field_h) // stride + 1
    out_w = (W + 2 * padding - field_w) // stride + 1

    i0 = np.repeat(np.arange(field_h), field_w)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_h), out_w)

    j0 = np.tile(np.arange(field_w), field_h)
    j0 = np.tile(j0, C)
    j1 = stride * np.tile(np.arange(out_w), out_h)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_h * field_w).reshape(-1, 1)

    return (k, i, j, out_h, out_w)

def im2col_indices(x, field_h, field_w, padding, stride):
    N, C, H, W = x.shape
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='constant'
    )
    k, i, j, out_h, out_w = get_im2col_indices(x.shape, field_h, field_w, padding, stride)
    cols = x_padded[:, k, i, j]  # (N, C*field_h*field_w, out_h*out_w)
    cols = cols.transpose(1, 2, 0).reshape(C * field_h * field_w, -1)
    return cols, out_h, out_w

def col2im_indices(cols, x_shape, field_h, field_w, padding, stride, out_h, out_w):
    N, C, H, W = x_shape
    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

    k, i, j, _, _ = get_im2col_indices(x_shape, field_h, field_w, padding, stride)
    cols_reshaped = cols.reshape(C * field_h * field_w, out_h * out_w, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

# -----------------------------
# 레이어들
# -----------------------------
class ReLU:
    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask
    def backward(self, grad_out):
        return grad_out * self.mask

class Flatten:
    def forward(self, x):
        self.in_shape = x.shape  # (N, C, H, W)
        return x.reshape(x.shape[0], -1)
    def backward(self, grad_out):
        return grad_out.reshape(self.in_shape)

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim), dtype=np.float32)
    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    def backward(self, grad_out, lr):
        dW = self.x.T @ grad_out
        db = np.sum(grad_out, axis=0, keepdims=True)
        dx = grad_out @ self.W.T
        # SGD 업데이트
        self.W -= lr * dW
        self.b -= lr * db
        return dx

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.C_in = in_channels
        self.C_out = out_channels
        self.K = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        # He init
        self.W = (np.random.randn(out_channels, in_channels, self.K, self.K).astype(np.float32)
                  * np.sqrt(2.0 / (in_channels * self.K * self.K)))
        self.b = np.zeros((out_channels, 1), dtype=np.float32)

    def forward(self, x):
        # x: (N, C, H, W)
        self.x_shape = x.shape
        self.cols, self.out_h, self.out_w = im2col_indices(x, self.K, self.K, self.padding, self.stride)
        self.W_col = self.W.reshape(self.C_out, -1)  # (C_out, C_in*K*K)

        out = self.W_col @ self.cols + self.b  # (C_out, N*out_h*out_w)
        N = x.shape[0]
        out = out.reshape(self.C_out, self.out_h, self.out_w, N).transpose(3, 0, 1, 2)  # (N,C_out,H',W')
        return out

    def backward(self, grad_out, lr):
        # grad_out: (N, C_out, out_h, out_w)
        N = grad_out.shape[0]
        grad_out_reshaped = grad_out.transpose(1, 2, 3, 0).reshape(self.C_out, -1)  # (C_out, N*out_h*out_w)

        dW_col = grad_out_reshaped @ self.cols.T  # (C_out, C_in*K*K)
        dW = dW_col.reshape(self.W.shape)
        db = np.sum(grad_out_reshaped, axis=1, keepdims=True)  # (C_out,1)

        dcols = self.W_col.T @ grad_out_reshaped  # (C_in*K*K, N*out_h*out_w)
        dx = col2im_indices(dcols, self.x_shape, self.K, self.K, self.padding, self.stride,
                            self.out_h, self.out_w)

        # SGD
        self.W -= lr * dW
        self.b -= lr * db
        # 메모리 피크 완화
        self.cols = None
        self.W_col = None
        return dx

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.P = int(pool_size)
        self.stride = int(stride)
        # 이 구현은 stride == pool_size, padding == 0 가정
        assert self.stride == self.P, "MaxPool2D: stride는 pool_size와 같아야 합니다."

    def forward(self, x):
        # x: (N, C, H, W)
        self.x_shape = x.shape
        N, C, H, W = x.shape
        P = self.P
        assert H % P == 0 and W % P == 0, f"입력 H,W는 pool_size({P})의 배수여야 합니다."

        # (N, C, H//P, P, W//P, P)로 블록화 → 각 2x2(또는 PxP) 내 최댓값
        x_resh = x.reshape(N, C, H // P, P, W // P, P)
        # (N, C, H//P, W//P, P*P) 로 평탄화해서 argmax 저장
        self.flat = x_resh.reshape(N, C, H // P, W // P, P * P)
        self.argmax = np.argmax(self.flat, axis=4)
        out = self.flat.max(axis=4)  # (N, C, H//P, W//P)
        return out

    def backward(self, grad_out):
        # grad_out: (N, C, H//P, W//P)
        N, C, H, W = self.x_shape
        P = self.P
        grad_flat = np.zeros_like(self.flat, dtype=grad_out.dtype)  # (N,C,H//P,W//P,P*P)

        # argmax 위치로 그라디언트 라우팅
        n, c, i, j = np.indices(self.argmax.shape)
        grad_flat[n, c, i, j, self.argmax] = grad_out

        # 원래 (N, C, H, W)로 되돌리기
        grad_resh = grad_flat.reshape(N, C, H // P, P, W // P, P)
        dx = grad_resh.reshape(N, C, H, W)

        # 메모리 해제
        self.flat = None
        self.argmax = None
        return dx

# -----------------------------
# CNN 네트워크
# -----------------------------
class CNNNetwork:
    def __init__(self, learning_rate=1e-3):
        self.lr = learning_rate
        self.conv1 = Conv2D(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(pool_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(pool_size=2, stride=2)

        self.flat = Flatten()
        self.fc1 = Linear(32 * (img_h // 4) * (img_w // 4), 256)  # 64->32->16
        self.relu3 = ReLU()
        self.fc2 = Linear(256, output_size)

    def forward(self, x, training=True):
        # x: (N, 4096) -> (N,1,64,64)
        x = x.reshape(-1, 1, img_h, img_w)
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.flat.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        logits = self.fc2.forward(x)

        probs = softmax(logits)
        self.last_logits = logits
        self.last_probs = probs
        return probs

    def backward(self, y_true):
        N = y_true.shape[0]
        dlogits = (self.last_probs - y_true) / N  # softmax+CE 그라디언트

        dout = self.fc2.backward(dlogits, self.lr)
        dout = self.relu3.backward(dout)
        dout = self.fc1.backward(dout, self.lr)

        dout = self.flat.backward(dout)

        dout = self.pool2.backward(dout)
        dout = self.relu2.backward(dout)
        dout = self.conv2.backward(dout, self.lr)

        dout = self.pool1.backward(dout)
        dout = self.relu1.backward(dout)
        _ = self.conv1.backward(dout, self.lr)
        return

    def train(self, X, Y, epochs, lr, batch_size=64, X_val=None, Y_val=None, patience=10):
        self.lr = lr
        best_acc = -1.0
        no_improve = 0

        for epoch in range(epochs):
            total_loss_sum, total_count = 0.0, 0
            correct = 0

            # ----- train loop -----
            for xb, yb in iterate_minibatches(X, Y, batch_size, shuffle=True):
                out = self.forward(xb, training=True)
                loss = cross_entropy(yb, out)
                total_loss_sum += loss * len(xb)
                total_count += len(xb)

                pred = np.argmax(out, axis=1)
                true = np.argmax(yb, axis=1)
                correct += np.sum(pred == true)

                self.backward(yb)

            avg_loss = total_loss_sum / max(total_count, 1)
            acc = 100.0 * correct / max(total_count, 1)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {acc:.2f}%")

            # ----- validation / early stop -----
            if (epoch + 1) % 5 == 0:
                # 검증셋 없으면 학습셋으로 대체(임시)
                valX, valY = (X, Y) if (X_val is None or Y_val is None) else (X_val, Y_val)
                val_loss, val_acc = self.evaluate(valX, valY, batch_size=256)

                if val_acc > best_acc:
                    best_acc = val_acc
                    self.save("best_cnn.pkl")
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print("[EarlyStop] no improvement")
                        break

            # ----- step LR decay (예: 10, 20 에폭에 반감) -----
            if (epoch + 1) in (10, 20):
                self.lr *= 0.5
                print(f"[LR] decayed to {self.lr:.6f}")


    def evaluate(self, X, Y, batch_size=256):
        total_loss_sum, total_count = 0, 0
        correct = 0

        for xb, yb in iterate_minibatches(X, Y, batch_size, shuffle=False):
            out = self.forward(xb, training=False)   # <-- 중요: training=False
            loss = cross_entropy(yb, out)
            total_loss_sum += loss * len(xb)
            total_count    += len(xb)

            pred = np.argmax(out, axis=1)
            true = np.argmax(yb, axis=1)
            correct += np.sum(pred == true)

        avg_loss = total_loss_sum / max(total_count, 1)
        acc      = correct / max(total_count, 1)
        print(f"Test Loss: {avg_loss:.6f}, Test Accuracy: {acc * 100:.2f}%")
        return avg_loss, acc


    def save(self, filepath="saved_cnn.pkl"):
        params = {
            "conv1_W": self.conv1.W, "conv1_b": self.conv1.b,
            "conv2_W": self.conv2.W, "conv2_b": self.conv2.b,
            "fc1_W": self.fc1.W, "fc1_b": self.fc1.b,
            "fc2_W": self.fc2.W, "fc2_b": self.fc2.b,
        }
        with open(filepath, "wb") as f:
            pickle.dump(params, f)
        print(f"✅ CNN model saved to {filepath}")

    def load(self, filepath="saved_cnn.pkl"):
        with open(filepath, "rb") as f:
            p = pickle.load(f)
        self.conv1.W, self.conv1.b = p["conv1_W"], p["conv1_b"]
        self.conv2.W, self.conv2.b = p["conv2_W"], p["conv2_b"]
        self.fc1.W, self.fc1.b = p["fc1_W"], p["fc1_b"]
        self.fc2.W, self.fc2.b = p["fc2_W"], p["fc2_b"]
        print(f"✅ CNN model loaded from {filepath}")

# -----------------------------
# 시각화
# -----------------------------
def visualize_predictions_and_true_images_with_predicted_images(model, test_data, test_labels, num_samples=10):
    indices = random.sample(range(len(test_data)), num_samples)
    sampled_data = test_data[indices]
    sampled_labels = test_labels[indices]

    predictions = model.forward(sampled_data, training=False)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(sampled_labels, axis=1)

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 5))
    for i, idx in enumerate(indices):
        pred_class = predicted_classes[i]
        pred_indices = np.where(np.argmax(test_labels, axis=1) == pred_class)[0]
        if len(pred_indices) > 0:
            pred_image = test_data[random.choice(pred_indices)]
            axes[0, i].imshow(pred_image.reshape(64, 64), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Pred: {pred_class}", color="green" if pred_class == true_classes[i] else "red")

        axes[1, i].imshow(sampled_data[i].reshape(64, 64), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"True: {true_classes[i]}")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 실행
# -----------------------------
if __name__ == "__main__":
    # 데이터 로드
    train_data, train_labels = load_data('train', 'train_data.csv', output_size)
    test_data, test_labels = load_data('test', 'test_data.csv', output_size)

    mean = train_data.mean()
    std  = train_data.std() + 1e-6
    train_data = (train_data - mean) / std
    test_data  = (test_data  - mean) / std
    print(f"[Normalize] mean={mean:.4f}, std={std:.4f}")
    # CNN 모델
    cnn = CNNNetwork(learning_rate=learning_rate)

    # 학습
    cnn.train(train_data, train_labels, epochs=epochs, lr=learning_rate, batch_size=batch_size)

    # 평가
    cnn.evaluate(test_data, test_labels)

    # 저장/로드/시각화
    cnn.save("saved_cnn.pkl")
    cnn.load("saved_cnn.pkl")
    visualize_predictions_and_true_images_with_predicted_images(cnn, test_data, test_labels, num_samples=10)
# update