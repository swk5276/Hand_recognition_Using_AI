import csv
import numpy as np
from pathlib import Path
from .config import IMG_H, IMG_W, OUTPUT_SIZE, DATA_DIR, NORM_STATS_PATH

def load_csv_dataset(sub_dir: str, file_name: str, output_size: int = OUTPUT_SIZE):
    """
    CSV 포맷: label, p0, p1, ..., p4095
    - 픽셀은 0..255 -> 0..1 로 스케일
    - 라벨이 1..OUTPUT_SIZE이면 0..OUTPUT_SIZE-1로 변환
    """
    file_path = DATA_DIR / sub_dir / file_name
    print(f"[load_csv_dataset] {file_path}")

    X_list, y_list = [], []
    # BOM/윈도우 대비: encoding='utf-8-sig', newline='' 권장
    with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            # 끝에 트레일링 쉼표로 생기는 빈 칸만 안전하게 제거
            if row[-1].strip() == "":
                row = row[:-1]
            # 중간에 빈 칸이 섞이면 데이터가 깨진 것 → 명확히 에러
            if any(v.strip() == "" for v in row):
                raise ValueError(f"Empty field detected in row; file may be corrupted: {file_path}")

            try:
                label_raw = int(row[0].strip())
            except Exception as e:
                raise ValueError(f"Label parse error at row head: {row[:3]} ({e})")

            # 픽셀
            try:
                pixels = np.array([float(v) for v in row[1:]], dtype=np.float32) / 255.0
            except Exception as e:
                raise ValueError(f"Pixel parse error. Row length={len(row)} ({e})")

            if pixels.size != IMG_H * IMG_W:
                raise ValueError(f"Expected {IMG_H*IMG_W} pixels but got {pixels.size} at {file_path}")

            X_list.append(pixels)
            y_list.append(label_raw)

    # 라벨 0-index/1-index 자동 정규화
    y_arr = np.array(y_list, dtype=np.int32)
    if y_arr.min() == 1 and y_arr.max() == output_size:
        y_arr = y_arr - 1

    if y_arr.min() < 0 or y_arr.max() >= output_size:
        raise ValueError(f"Label out of range after normalization: min={y_arr.min()}, max={y_arr.max()}, output_size={output_size}")

    # 원-핫
    Y = np.zeros((len(y_arr), output_size), dtype=np.float32)
    Y[np.arange(len(y_arr)), y_arr] = 1.0

    X = np.array(X_list, dtype=np.float32)
    return X, Y


def compute_and_save_norm_stats(train_X: np.ndarray, path: Path | None = None):
    """
    train_X 기준 mean/std를 계산해 npz로 저장.
    기본 경로는 NORM_STATS_PATH.
    """
    if path is None:
        path = NORM_STATS_PATH
    path = Path(path)

    mean = float(train_X.mean())
    std  = float(train_X.std() + 1e-6)  # zero-div 방지

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, mean=mean, std=std)
    print(f"[norm] saved mean={mean:.4f}, std={std:.4f} to {path}")
    return mean, std


def load_norm_stats(path: Path | None = None):
    """
    npz에서 mean/std 로드. 키 누락/파일 없음에 대해 명확한 에러 메시지 제공.
    """
    if path is None:
        path = NORM_STATS_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Normalization file not found: {path}\n"
            f"→ Run your training/precompute step to generate it (compute_and_save_norm_stats)."
        )
    try:
        d = np.load(path)
        mean = float(d["mean"])
        std  = float(d["std"])
    except KeyError as e:
        raise KeyError(
            f"Key {e} not found in {path}. The file may be corrupted or created with wrong keys.\n"
            f"Expected keys: 'mean', 'std'. Recompute the file."
        )
    return mean, std


def apply_norm(X: np.ndarray, mean: float, std: float):
    return (X - mean) / std


def iterate_minibatches(X, Y, batch_size=64, shuffle=True):
    """
    셔플/미니배치 유틸. 학습 안정성을 위해 shuffle=True 권장.
    """
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        j = idx[start:start+batch_size]
        if j.size:
            yield X[j], Y[j]
