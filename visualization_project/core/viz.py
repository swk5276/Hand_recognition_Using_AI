# visualization_project/core/viz.py
import io, base64, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import IMG_H, IMG_W

def draw_pred_vs_true_grid(model, test_data, test_labels, num_samples=10, title=""):
    # 무작위 샘플
    indices = random.sample(range(len(test_data)), num_samples)
    sampled_data = test_data[indices]
    sampled_labels = test_labels[indices]

    # 모델 추론
    predictions = model.forward(sampled_data, training=False)
    predicted = np.argmax(predictions, axis=1)
    true = np.argmax(sampled_labels, axis=1)

    # 시각화
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 5))
    if title:
        fig.suptitle(title)

    for i in range(num_samples):
        # True image
        axes[1, i].imshow(sampled_data[i].reshape(IMG_H, IMG_W), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title(f"True: {true[i]}")

        # One example image predicted to the same class
        pred_class = predicted[i]
        pred_indices = np.where(np.argmax(test_labels, axis=1) == pred_class)[0]
        if len(pred_indices) > 0:
            pred_image = test_data[random.choice(pred_indices)]
            axes[0, i].imshow(pred_image.reshape(IMG_H, IMG_W), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"Pred: {pred_class}",
            color=("green" if pred_class == true[i] else "red")
        )

    plt.tight_layout()

    # PNG Base64 인코딩
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()
    plt.close(fig)
    return image_base64
