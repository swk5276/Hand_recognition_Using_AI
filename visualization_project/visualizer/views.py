# visualizer/views.py
from django.shortcuts import render, HttpResponse
import io, base64, random, logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.config import OUTPUT_SIZE, NORM_STATS_PATH, MLP_WEIGHTS_PATH, CNN_WEIGHTS_PATH
from core.mlp_numpy import NeuralNetwork as MLP, load_data as load_mlp_data
from core.cnn_numpy import CNNNetwork as CNN

logger = logging.getLogger(__name__)

def _img_to_base64(img2d, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6))
    ax.imshow(img2d, cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close(fig)
    return b64

def visualize_single(request):
    try:
        # --- 데이터 로드 & 정규화 ---
        X_test, Y_test = load_mlp_data("test", "test_data.csv", OUTPUT_SIZE)
        stats = np.load(NORM_STATS_PATH)
        mean, std = stats["mean"], stats["std"]
        Xn = (X_test - mean) / std

        # --- 인덱스 결정 (GET ?idx=) ---
        if "idx" in request.GET:
            try:
                idx = int(request.GET["idx"])
            except ValueError:
                idx = random.randrange(len(Xn))
            idx = max(0, min(idx, len(Xn)-1))
        else:
            idx = random.randrange(len(Xn))

        x = Xn[idx:idx+1]                    # (1, 4096)
        true = int(np.argmax(Y_test[idx]))   # 정답 라벨

        # --- 모델 로드 ---
        mlp = MLP();  mlp.load(str(MLP_WEIGHTS_PATH))
        cnn = CNN();  cnn.load(str(CNN_WEIGHTS_PATH))

        # --- 예측 ---
        mlp_probs = mlp.forward(x, training=False).reshape(-1)
        cnn_probs = cnn.forward(x, training=False).reshape(-1)

        mlp_pred = int(np.argmax(mlp_probs))
        cnn_pred = int(np.argmax(cnn_probs))

        # --- 원본(정규화 전) 샘플 이미지 ---
        true_img_b64 = _img_to_base64(
            X_test[idx].reshape(64, 64),
            title=f"Index {idx} • True: {true}"
        )

        # --- 예측 클래스 대표 이미지(테스트셋에서 같은 라벨 중 1장 선택) ---
        def pick_pred_image_base64(pred_label, title_prefix):
            matches = np.where(np.argmax(Y_test, axis=1) == pred_label)[0]
            if matches.size == 0:
                return None
            pick = int(random.choice(matches))
            return _img_to_base64(
                X_test[pick].reshape(64, 64),
                title=f"{title_prefix} sample of label {pred_label} (idx {pick})"
            )

        mlp_pred_img_b64 = pick_pred_image_base64(mlp_pred, "MLP")
        cnn_pred_img_b64 = pick_pred_image_base64(cnn_pred, "CNN")

        # --- Top-5 표를 위해 상위 확률 추출(옵션) ---
        mlp_top5_idx = np.argsort(mlp_probs)[-5:][::-1]
        cnn_top5_idx = np.argsort(cnn_probs)[-5:][::-1]
        mlp_top5 = [(int(i), float(mlp_probs[i])) for i in mlp_top5_idx]
        cnn_top5 = [(int(i), float(cnn_probs[i])) for i in cnn_top5_idx]

        ctx = {
            "idx": idx,
            "image_base64": true_img_b64,          # 원본/정답 이미지
            "true_label": true,

            "mlp_pred": mlp_pred,
            "cnn_pred": cnn_pred,
            "mlp_top5": mlp_top5,
            "cnn_top5": cnn_top5,
            "same": (mlp_pred == cnn_pred),

            # ✅ 추가: 예측 클래스 대표 이미지
            "mlp_pred_image_base64": mlp_pred_img_b64,
            "cnn_pred_image_base64": cnn_pred_img_b64,
        }
        return render(request, "visualizer/visualize_single.html", ctx)

    except Exception as e:
        logger.exception("visualize_single failed")
        return HttpResponse(f"Error: {e}", status=500)


def _predict_and_draw(model, title: str):
    # 데이터 로드
    test_X, test_Y = load_csv_dataset('test', 'test_data.csv')
    # 정규화 통계 로드 후 적용
    mean, std = load_norm_stats()
    test_X = apply_norm(test_X, mean, std)

    # 이미지 Base64 생성
    img_b64 = draw_pred_vs_true_grid(model, test_X, test_Y, num_samples=10, title=title)
    return img_b64

def visualize_mlp(request):
    try:
        model = NeuralNetwork()
        if not MLP_WEIGHTS_PATH.exists():
            return HttpResponse(f"MLP 모델 파일이 없습니다: {MLP_WEIGHTS_PATH}", status=500)
        model.load(str(MLP_WEIGHTS_PATH))
        image_base64 = _predict_and_draw(model, title="MLP Predictions")
        return render(request, "visualizer/visualize.html", {"image_base64": image_base64, "title": "MLP"})
    except Exception as e:
        logger.exception("visualize_mlp failed")
        return HttpResponse(f"Error: {e}", status=500)

def visualize_cnn(request):
    try:
        model = CNNNetwork()
        if not CNN_WEIGHTS_PATH.exists():
            return HttpResponse(f"CNN 모델 파일이 없습니다: {CNN_WEIGHTS_PATH}", status=500)
        model.load(str(CNN_WEIGHTS_PATH))
        image_base64 = _predict_and_draw(model, title="CNN Predictions")
        return render(request, "visualizer/visualize.html", {"image_base64": image_base64, "title": "CNN"})
    except Exception as e:
        logger.exception("visualize_cnn failed")
        return HttpResponse(f"Error: {e}", status=500)
