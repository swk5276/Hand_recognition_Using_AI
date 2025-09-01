# visualization_project/core/config.py
from pathlib import Path
import numpy as np

# -----------------------------
# 하이퍼파라미터 (공유)
# -----------------------------
IMG_H, IMG_W, IN_CHANNELS = 64, 64, 1
OUTPUT_SIZE = 111
LEARNING_RATE = 1e-3
EPOCHS = 30          # MLP는 더 길게 돌리고 싶다면 뷰/학습 스크립트에서 override 가능
BATCH_SIZE = 64
SEED = 42
np.random.seed(SEED)

# -----------------------------
# 경로 (공유)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # visualization_project/
DATA_DIR = BASE_DIR / "DATA"
MODEL_DIR = BASE_DIR / "core"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 현재 운영 중인 파일명에 맞춤
MLP_WEIGHTS_PATH = MODEL_DIR / "saved_model.pkl"   # ← 기존에 이미 있는 MLP 파일명
CNN_WEIGHTS_PATH = MODEL_DIR / "saved_cnn.pkl"
NORM_STATS_PATH  = MODEL_DIR / "saved_model.npz"   # ← 방금 저장/사용한 통계 파일명

#python manage.py runserver 127.0.0.1:8000