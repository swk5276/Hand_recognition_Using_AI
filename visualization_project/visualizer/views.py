from django.shortcuts import render
import random
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from core.Neural_Network import NeuralNetwork, load_data
import logging

logger = logging.getLogger(__name__)

def visualize_predictions(request):
    logger.info("Starting visualize_predictions view.")
    # 신경망 하이퍼파라미터 설정
    input_size = 4096
    hidden_layers = [256, 240, 220]
    output_size = 111
    learning_rate = 0.0001
    epochs = 1

    try:
        # 데이터 로드 (함수 내부에서 실행)
        train_data, train_labels = load_data('train', 'train_data.csv', output_size)
        test_data, test_labels = load_data('test', 'test_data.csv', output_size)

        # 신경망 객체 생성 및 학습
        nn = NeuralNetwork(input_size, hidden_layers, output_size, learning_rate)
        nn.train(train_data, train_labels, epochs, learning_rate)
        nn.evaluate(test_data, test_labels)

        # 시각화
        indices = random.sample(range(len(test_data)), 10)
        sampled_data = test_data[indices]
        sampled_labels = test_labels[indices]
        predictions = nn.forward(sampled_data, training=False)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(sampled_labels, axis=1)

        # 그래프 생성
        fig, axes = plt.subplots(2, 10, figsize=(20, 5))

        for i in range(10):
            # 실제 이미지
            axes[1, i].imshow(sampled_data[i].reshape(64, 64), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"True: {true_classes[i]}")

            # 예측된 클래스에 해당하는 이미지 찾기
            pred_class = predicted_classes[i]
            pred_indices = np.where(np.argmax(test_labels, axis=1) == pred_class)[0]
            if len(pred_indices) > 0:
                pred_image = test_data[pred_indices[0]]
                axes[0, i].imshow(pred_image.reshape(64, 64), cmap='gray')
                axes[0, i].axis('off')
                axes[0, i].set_title(f"Pred: {pred_class}", color="green" if pred_class == true_classes[i] else "red")
            else:
                # 예측된 클래스에 해당하는 데이터가 없는 경우
                axes[0, i].axis('off')
                axes[0, i].set_title(f"Pred: {pred_class} (No Match)", color="red")

        plt.tight_layout()

        # 그래프를 메모리 버퍼로 저장
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        # Base64로 인코딩
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()

        # HTML로 렌더링
        return render(request, 'visualizer/visualize.html', {'image_base64': image_base64})
    except Exception as e:
        return render(request, 'visualizer/error.html', {'error_message': str(e)})
