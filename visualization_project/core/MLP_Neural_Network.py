import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os
from pathlib import Path
import pickle


# 신경망 하이퍼파라미터 설정
input_size = 4096              # 입력 크기: 64x64 이미지의 총 픽셀 수 (4096)
hidden_layers = [256,196,128]  # 은닉층 노드 수
output_size = 111              # 출력층 노드 수: 분류할 클래스 수
learning_rate = 0.0001         # 학습률: 가중치 업데이트의 크기
epochs = 100                  # 전체 학습 데이터 반복 횟수

#데이터 로드 함수 
def load_data(sub_dir, file_name, output_size):
    BASE_DIR = Path(__file__).resolve().parent.parent  # visualization_project 기준의 경로
    file_path = BASE_DIR / 'DATA' / sub_dir / file_name
    print(f"Loading file from: {file_path}")

    pixel_data = [] # 픽셀 데이터
    labels = [] # 레이블(정답) 데이터

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 문자열을 정수로 변환
            label = int(row[0].strip()) - 1  # 첫 번째 열은 레이블 (0~110)
            row_data = [float(value) / 255.0 for value in row[1:]]  # 픽셀 데이터 정규화 (0~255 -> 0~1)
            pixel_data.append(row_data)  # 픽셀 데이터를 리스트에 추가

            # 원핫 인코딩 생성
            one_hot_label = [0] * output_size  # 출력 크기만큼 0으로 채운 리스트 생성
            one_hot_label[label] = 1  # 해당 레이블 인덱스를 1로 설정
            labels.append(one_hot_label)  # 원핫 인코딩된 레이블을 리스트에 추가

    # 데이터를 numpy 배열로 변환하여 반환
    return np.array(pixel_data), np.array(labels)


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

#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 시그모이드 함수: 1 / (1 + e^(-x))

# Relu(순전파 사용)
def relu(x):
    return np.maximum(0, x)  # ReLU 함수: 입력이 음수이면 0, 양수이면 그대로 출력

# tanh(순전파 사용)
def tanh(x):
    return np.tanh(x)  # tanh 함수: (e^x - e^(-x)) / (e^x + e^(-x))

#시그모이드 미분(역전파 사용)
def sigmoid_derivative(x):  
    return sigmoid(x) * (1 - sigmoid(x))  # 시그모이드 미분: sigmoid(x) * (1 - sigmoid(x))

# Relu 미분(역전파 사용)
def relu_derivative(x):
    return np.where(x > 0, 1, 0)  # ReLU 미분: 입력이 양수이면 1, 음수이면 0

# tanh 미분(역전파 사용)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2  # tanh 미분: 1 - tanh(x)^2

# 출력층 활성화함수 정의
def softmax(x):
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # softmax: 확률로 변환

"손실함수 정의"

# 평균제곱오차
def MSE(y_true, y_pred):    # 실제값과 예측값을 사용하여 평균제곱오차 계산
    return np.mean((y_true-y_pred) ** 2)

def cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

"정확도 계산"

# 정확도 계산 함수 정의
def accuracy(y_true, y_pred):   # 정답과 예측값을 받아서 정확도 계산
    if y_true.ndim > 1:      
        y_true = np.argmax(y_true, axis=1)  #원한 인코딩 정수 변환
    if y_pred.ndim > 1:  
        y_pred = np.argmax(y_pred, axis=1)  # 원핫 인코딩 정수 변환
    correct = np.sum(y_true == y_pred)  # 정답과 예측이 일치하는 개수 계산
    return correct / len(y_true)  # 정확도 계산

# 신경망 클래스 정의
class NeuralNetwork:
    def __init__(
        self,
        input_size=4096,
        hidden_layers=[256,196,128],
        output_size=111,
        learning_rate=0.0001
    ):
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.weights, self.biases = [], []
        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    # 순전파
    def forward(self, x, training=True):
        self.activations = [x]  # 입력 데이터를 첫 번째 활성화 값으로 설정
        # 은닉층 활성화 함수(Relu) 계산
        for i in range(len(self.weights) - 1):  # 은닉층 수만큼 반복
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]  # 업데이트 된 가중치와 편향을 통하여 활성화 함수 계산 
            a = relu(z)  # 활성화 함수 적용
            self.activations.append(a)
        # 출력층 활성화 함수(소프트 맥스)계산
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = softmax(z)  # 소프트맥스 활성화 함수 적용
        self.activations.append(a)
        return self.activations[-1] # 출력층 활성화 값 반환

    # 역전파
    def backward(self, y_true):
        # 출력층 오차 (softmax+cross_entropy)
        deltas = [self.activations[-1] - y_true]
        
        # 은닉층에서의 오차 전파(출력층-> 은닉층 , 은닉층->입력층)
        for i in reversed(range(len(self.weights) - 1)):    
            delta = deltas[-1].dot(self.weights[i + 1].T) * relu_derivative(self.activations[i + 1])    # 은닉층 오차 계산
            deltas.append(delta)
        deltas.reverse()  # 순서를 원래대로 변경
        
        # 가중치 및 편향 업데이트
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.activations[i].T.dot(deltas[i])    # 가중치 업데이트  
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) # 편향 업데이트


    # 학습
    def train(self, x, y, epochs, learning_rate, batch_size=64):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            total_loss_sum, total_count = 0.0, 0
            correct = 0

            for xb, yb in iterate_minibatches(x, y, batch_size, shuffle=True):
                out = self.forward(xb, training=True)
                loss = cross_entropy(yb, out)
                total_loss_sum += loss * len(xb)
                total_count    += len(xb)

                pred = np.argmax(out, axis=1)
                true = np.argmax(yb, axis=1)
                correct += np.sum(pred == true)

                self.backward(yb)  # (아래 L2/Dropout을 추가했다면 그에 맞춰 수정)

            avg_loss = total_loss_sum / max(total_count, 1)
            acc = 100.0 * correct / max(total_count, 1)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Train Accuracy: {acc:.2f}%")

            if (epoch + 1) % 10 == 0:
                print("모델 평가")
                self.evaluate(x, y)  # 실제로는 val로 분리 권장

    # 평가 함수
    def evaluate(self, x, y):
        test_output = self.forward(x, training=False)   #테스트 데이터에 대한 예측값
        test_loss = cross_entropy(y,test_output) #테스트 데이터에 대한 손실 계산
        test_predictions = np.argmax(test_output, axis=1)   #테스트 데이터 예측값
        test_true_labels = np.argmax(y, axis=1) #테스트 데이터 실제값
        test_accuracy = accuracy(test_true_labels, test_predictions)    #테스트 데이터 정확도 계산

        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_loss, test_accuracy

    def save(self, filepath="saved_model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump({"weights": self.weights, "biases": self.biases}, f)
        print(f"✅ Model saved to {filepath}")


    def load(self, filepath="saved_model.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.weights = data["weights"]
        self.biases = data["biases"]
        print(f"✅ Model loaded from {filepath}")


#시각화
# 시각화 함수 (인덱스 버그 수정 + 안전 처리)
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


if __name__ == "__main__":    
    # 데이터 로드
    train_data, train_labels = load_data('train', 'train_data.csv', output_size)
    test_data, test_labels = load_data('test', 'test_data.csv', output_size)

    # 신경망 객체 생성
    nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)

    # 학습 실행
    nn.train(train_data, train_labels, epochs, learning_rate)

    # 테스트 데이터로 평가
    nn.evaluate(test_data, test_labels)

    nn.save("saved_model.pkl")
    # 시각화 함수 호출

    nn.load("saved_model.pkl")
    visualize_predictions_and_true_images_with_predicted_images(nn, test_data, test_labels)