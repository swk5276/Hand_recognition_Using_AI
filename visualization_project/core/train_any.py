# core/train_any.py
import argparse
import os
from pathlib import Path
import numpy as np

# 공통 설정/경로
from core.config import (
    OUTPUT_SIZE,
    LEARNING_RATE, EPOCHS, BATCH_SIZE,
    MLP_WEIGHTS_PATH, CNN_WEIGHTS_PATH, NORM_STATS_PATH,
)

# 모델 & 데이터 로더
# - 데이터 로드는 MLP 쪽 load_data를 사용 (64x64 CSV → X(0~1), Y(one-hot))
from core.mlp_numpy import NeuralNetwork as MLPModel, load_data as load_data_mlp
from core.cnn_numpy import CNNNetwork as CNNModel

# 정규화 유틸: 없으면 생성, 있으면 재사용
def _save_norm_stats(train_X, path: Path):
    mean = float(train_X.mean())
    std = float(train_X.std() + 1e-6)
    np.savez(path, mean=mean, std=std)
    print(f"✅ Saved norm stats to {path} (mean={mean:.6f}, std={std:.6f})")
    return mean, std

def _load_norm_stats_or_compute(train_X, path: Path):
    if path.exists():
        try:
            stats = np.load(path)
            mean, std = float(stats["mean"]), float(stats["std"])
            print(f"ℹ️  Using existing norm stats at {path} (mean={mean:.6f}, std={std:.6f})")
            return mean, std
        except Exception as e:
            print(f"⚠️  Failed to read {path} ({e}). Recomputing norm stats...")
    # 없거나 읽기 실패 → 새로 저장
    return _save_norm_stats(train_X, path)

def _normalize(X, mean, std):
    return (X - mean) / std

def train_mlp(force: bool = False):
    """
    - saved_mlp.pkl 이 존재하고 force=False면 학습을 건너뜁니다.
    - norm_stats.npz가 없으면 학습 데이터로 새로 계산하여 저장합니다.
    """
    print("\n====== [MLP] ======")
    if MLP_WEIGHTS_PATH.exists() and not force:
        print(f"ℹ️  Found existing MLP weights: {MLP_WEIGHTS_PATH} (skip training).")
        # 그래도 norm stats가 없으면 만들어 둠
        Xtr, Ytr = load_data_mlp('train', 'train_data.csv', OUTPUT_SIZE)
        _load_norm_stats_or_compute(Xtr, NORM_STATS_PATH)
        return

    # 데이터 로드
    Xtr, Ytr = load_data_mlp('train', 'train_data.csv', OUTPUT_SIZE)
    Xte, Yte = load_data_mlp('test',  'test_data.csv',  OUTPUT_SIZE)

    # 정규화 통계 준비(재사용/생성)
    mean, std = _load_norm_stats_or_compute(Xtr, NORM_STATS_PATH)

    # 정규화
    Xtr = _normalize(Xtr, mean, std)
    Xte = _normalize(Xte, mean, std)

    # 학습
    model = MLPModel()
    model.train(Xtr, Ytr, epochs=EPOCHS, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE)

    # 평가
    model.evaluate(Xte, Yte)

    # 저장
    model.save(str(MLP_WEIGHTS_PATH))

def train_cnn(force: bool = False):
    """
    - saved_cnn.pkl 이 존재하고 force=False면 학습을 건너뜁니다.
    - norm_stats.npz가 없으면 학습 데이터로 새로 계산하여 저장합니다.
    """
    print("\n====== [CNN] ======")
    if CNN_WEIGHTS_PATH.exists() and not force:
        print(f"ℹ️  Found existing CNN weights: {CNN_WEIGHTS_PATH} (skip training).")
        # 그래도 norm stats가 없으면 만들어 둠
        Xtr, Ytr = load_data_mlp('train', 'train_data.csv', OUTPUT_SIZE)
        _load_norm_stats_or_compute(Xtr, NORM_STATS_PATH)
        return

    # 데이터 로드 (CSV 포맷은 동일하므로 MLP의 로더 재사용)
    Xtr, Ytr = load_data_mlp('train', 'train_data.csv', OUTPUT_SIZE)
    Xte, Yte = load_data_mlp('test',  'test_data.csv',  OUTPUT_SIZE)

    # 정규화 통계 준비(재사용/생성)
    mean, std = _load_norm_stats_or_compute(Xtr, NORM_STATS_PATH)

    # 정규화
    Xtr = _normalize(Xtr, mean, std)
    Xte = _normalize(Xte, mean, std)

    # 학습
    model = CNNModel(learning_rate=LEARNING_RATE)
    model.train(Xtr, Ytr, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE)

    # 평가
    model.evaluate(Xte, Yte)

    # 저장
    model.save(str(CNN_WEIGHTS_PATH))

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP and/or CNN, save weights and norm_stats.npz (reusing existing files unless --force)."
    )
    parser.add_argument(
        "target",
        choices=["mlp", "cnn", "all"],
        help="Which model to train."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retrain even if saved weights exist."
    )
    args = parser.parse_args()

    # 모델 디렉토리 생성 보장
    for p in [MLP_WEIGHTS_PATH.parent, CNN_WEIGHTS_PATH.parent, NORM_STATS_PATH.parent]:
        p.mkdir(parents=True, exist_ok=True)

    if args.target in ("mlp", "all"):
        train_mlp(force=args.force)
    if args.target in ("cnn", "all"):
        train_cnn(force=args.force)

if __name__ == "__main__":
    main()


# # MLP만 (기존 가중치 있으면 스킵)
# python -m core.train_any mlp

# # CNN만 (기존 가중치 있으면 스킵)
# python -m core.train_any cnn

# # 둘 다 (기존 가중치 있으면 각각 스킵)
# python -m core.train_any all

# # 강제 재학습
# python -m core.train_any all --force
