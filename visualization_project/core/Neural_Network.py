import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import os
from pathlib import Path


# 신경망 하이퍼파라미터 설정
input_size = 4096              # 입력 크기: 64x64 이미지의 총 픽셀 수 (4096)
hidden_layers = [256,240,220]  # 은닉층 노드 수
output_size = 111              # 출력층 노드 수: 분류할 클래스 수
learning_rate = 0.0001         # 학습률: 가중치 업데이트의 크기
epochs = 1                   # 전체 학습 데이터 반복 횟수

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


"활성화함수 정의"

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
    exp_x = np.exp(x)  # 입력값 x에 대한 지수 함수 계산
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)  # softmax: 확률로 변환

"손실함수 정의"

# 평균제곱오차
def MSE(y_true, y_pred):    # 실제값과 예측값을 사용하여 평균제곱오차 계산
    return np.mean((y_true-y_pred) ** 2)

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
    def __init__(self, input_size, hidden_layers, output_size, learning_rate):
        self.learning_rate = learning_rate  
        self.weights = [] 
        self.biases = []  
        # 입력, 은닉, 출력층 노드 수
        layer_sizes = [input_size] + hidden_layers + [output_size]  
        
        # 각 레이어 가중치와 편향 초기화
        for i in range(len(layer_sizes) - 1):
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])  # 심층 신경망에서 기울기 소실 방지를 위한 he 초기화
            bias = np.zeros((1, layer_sizes[i + 1]))  # 편향 초기화
            self.weights.append(weight)
            self.biases.append(bias)

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
        # 출력층 오차
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
    def train(self, x, y, epochs, learning_rate):
        self.learning_rate = learning_rate

        # 클래스별 데이터 인덱스 생성
        class_indices = {label: np.where(np.argmax(y, axis=1) == label)[0] for label in range(y.shape[1])}

        for epoch in range(epochs): 
            if (epoch + 1) % 10 == 0:   #주기적 모델 평가
                print("모델 범용성 평가")
                self.evaluate(x, y) 

            total_loss = 0  # 총 손실 초기화
            correct_predictions = 0  # 정확한 예측 수 초기화

            # 클래스별 학습 루프
            for class_label, indices in class_indices.items():
                np.random.shuffle(indices)  # 클래스 데이터를 섞어서 순차적으로 학습
                class_x = x[indices]    #클래스 픽셀 데이터
                class_y = y[indices]    #클래스 레이블 데이터

                # 순전파
                output = self.forward(class_x, training=True)   
                loss = MSE(class_y, output) #실제값과 예측값의 손실 계산
                total_loss += loss  #총 손실 누적

                # 정확도 계산
                predictions = np.argmax(output, axis=1) #예측값
                true_labels = np.argmax(class_y, axis=1)    #실제값
                correct_predictions += np.sum(predictions == true_labels)   #정확한 예측 수 계산

                # 역전파
                self.backward(class_y)  

            # 평균 손실 및 학습 정확도 계산
            average_loss = total_loss / len(x)  # 전체 데이터 크기로 나눔
            train_accuracy = correct_predictions / len(x) * 100

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {average_loss}, Train Accuracy: {train_accuracy:.2f}%")

    # 평가 함수
    def evaluate(self, x, y):
        test_output = self.forward(x, training=False)   #테스트 데이터에 대한 예측값
        test_loss = MSE(y, test_output) #테스트 데이터에 대한 손실 계산
        test_predictions = np.argmax(test_output, axis=1)   #테스트 데이터 예측값
        test_true_labels = np.argmax(y, axis=1) #테스트 데이터 실제값
        test_accuracy = accuracy(test_true_labels, test_predictions)    #테스트 데이터 정확도 계산

        print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100:.2f}%")
        return test_loss, test_accuracy
    
#시각화
def visualize_predictions_and_true_images_with_predicted_images(model, test_data, test_labels, num_samples=10):

    # 테스트 데이터에서 무작위로 샘플 선택
    indices = random.sample(range(len(test_data)), num_samples)
    sampled_data = test_data[indices]
    sampled_labels = test_labels[indices]

    # 신경망 예측 수행
    predictions = model.forward(sampled_data, training=False)
    predicted_classes = np.argmax(predictions, axis=1)  # 예측된 클래스
    true_classes = np.argmax(sampled_labels, axis=1)    # 실제 클래스

    # 시각화 설정
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 5))

    for i, index in enumerate(indices):
        # Pred: 예측된 클래스에 해당하는 이미지를 찾음
        pred_class = predicted_classes[i]
        pred_image_index = np.argmax(np.argmax(test_labels, axis=1) == pred_class)
        pred_image = test_data[pred_image_index]

        # 예측 이미지 출력
        axes[0, i].imshow(pred_image.reshape(64, 64), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"Pred: {pred_class}", color="green" if pred_class == true_classes[i] else "red")

        # True: 실제 이미지 출력
        axes[1, i].imshow(sampled_data[i].reshape(64, 64), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"True: {true_classes[i]}")

    # 전체 레이아웃 조정
    fig.suptitle("Predicted vs. True Images", fontsize=16)
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
    # 시각화 함수 호출
    visualize_predictions_and_true_images_with_predicted_images(nn, test_data, test_labels)