from django.shortcuts import render, HttpResponse
import random, io, base64, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 서버 환경에서는 꼭 Agg 백엔드
import matplotlib.pyplot as plt
from visualization_project.core.MLP_Neural_Network import NeuralNetwork, load_data

logger = logging.getLogger(__name__)

# 프로젝트 루트 기준 절대경로 (visualization_project/)
BASE_DIR   = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "core" / "saved_model.pkl"   # ✅ core/saved_model.pkl 고정

# python manage.py runserver 127.0.0.1:8000

def visualize_predictions(request):
    logger.info("Starting visualize_predictions view.")
    try:
        # 테스트 데이터 로드 (추론용)
        test_data, test_labels = load_data('test', 'test_data.csv', output_size=111)

        # 신경망 객체 생성
        nn = NeuralNetwork()

        # 학습된 모델 로드
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
        nn.load(str(MODEL_PATH))

        # 무작위 샘플 추론
        indices = random.sample(range(len(test_data)), 10)
        sampled_data = test_data[indices]
        sampled_labels = test_labels[indices]
        predictions = nn.forward(sampled_data, training=False)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(sampled_labels, axis=1)

        # 시각화
        fig, axes = plt.subplots(2, 10, figsize=(20, 5))
        for i in range(10):
            # True 이미지
            axes[1, i].imshow(sampled_data[i].reshape(64, 64), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"True: {true_classes[i]}")

            # Pred 이미지
            pred_class = predicted_classes[i]
            pred_indices = np.where(np.argmax(test_labels, axis=1) == pred_class)[0]
            if len(pred_indices) > 0:
                pred_image = test_data[pred_indices[0]]
                axes[0, i].imshow(pred_image.reshape(64, 64), cmap='gray')
                axes[0, i].axis('off')
                axes[0, i].set_title(
                    f"Pred: {pred_class}",
                    color=("green" if pred_class == true_classes[i] else "red")
                )
            else:
                axes[0, i].axis('off')
                axes[0, i].set_title(f"Pred: {pred_class} (No Match)", color="red")

        plt.tight_layout()

        # Base64 인코딩
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close(fig)

        return render(request, 'visualizer/visualize.html', {'image_base64': image_base64})

    except Exception as e:
        logger.exception("visualize_predictions failed")
        return HttpResponse(f"Error: {e}", status=500)
