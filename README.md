<img width="1910" height="999" alt="image" src="https://github.com/user-attachments/assets/e2202162-c595-47d5-9dc6-5cebc5353f34" />

# Hand Recognition Using AI

NumPy 기반 CNN & MLP 직접 구현 + Django Web 시각화 시스템
딥러닝 프레임워크(TensorFlow/PyTorch) 없이 NumPy만으로 CNN과 MLP를 구현하고, Django 웹앱에서 예측 결과를 시각화하는 엔드투엔드 AI 시스템 프로젝트

## 프로젝트 개요

본 프로젝트는 **111종 손 이미지 분류(Classification)** 문제를 해결하기 위해 진행되었습니다.

* **Custom 모델 구현**
  CNN과 MLP를 모두 NumPy만으로 직접 구현 → forward, backward, optimizer까지 수동 작성
* **데이터 파이프라인 구축**
  CSV 기반 데이터셋 로더, 정규화(norm\_stats), 원핫 인코딩, 미니배치 생성기
* **성능 개선 기법 적용**
  Normalization, Dropout, Label Smoothing, Data Augmentation, Learning Rate Decay
* **웹 시각화**
  Django 기반 웹 서비스에서 MLP와 CNN 예측 결과를 비교 시각화

---

## 핵심 목표

### 1. Custom 모델 구현

* **CNN**

  * Conv2D (im2col / col2im 기반, stride & padding 지원)
  * MaxPool2D (argmax 기반 backward 구현)
  * Fully Connected + Softmax
* **MLP**

  * 다층 퍼셉트론: \[256 → 196 → 128] 은닉층
  * 활성화 함수: ReLU / Sigmoid / Tanh 지원
  * Softmax + Cross-Entropy Loss

### 2. 성능 향상 기법 적용

* 데이터 정규화 (mean/std 저장 후 재사용)
* He 초기화로 학습 안정화
* Regularization 기법: Dropout, Weight Decay
* Data Augmentation: Flip, Random Crop → 테스트 정확도 +17% 개선
* Early Stopping + Best Checkpoint 저장

### 3. 웹 기반 시각화

* 테스트셋 무작위 샘플 선택 후 MLP & CNN 각각 예측
* 실제 라벨과 비교하여 올바른 예측은 초록색, 틀린 예측은 빨간색
* 결과 이미지를 base64 인코딩하여 Django 템플릿에 직접 렌더링

---

## 사용 기술 스택

* Python 3.11
* NumPy (딥러닝 프레임워크 미사용)
* Matplotlib (결과 시각화 및 Django 출력용)
* Django 5.x (웹 서비스 프레임워크)
* CSV Dataset (train\_data.csv, test\_data.csv)

---

## 프로젝트 구조

```
visualization_project/
├── core/
│   ├── mlp_numpy.py         # MLP 모델 (NumPy 기반)
│   ├── cnn_numpy.py         # CNN 모델 (Conv/Pool/FC/Softmax)
│   ├── data.py              # 데이터 로딩, 정규화, 원핫 인코딩
│   ├── config.py            # 하이퍼파라미터 및 경로 설정
│   ├── train_any.py         # MLP & CNN 학습 / 저장 / norm_stats 관리
│   ├── saved_model.pkl      # 학습된 MLP 가중치
│   ├── saved_cnn.pkl        # 학습된 CNN 가중치
│   └── norm_stats.npz       # 정규화(mean/std)
├── visualizer/
│   ├── views.py             # Django 뷰 (예측 시각화)
│   ├── models.py            # 모델 로더
│   └── templates/visualizer/
│       └── visualize.html   # 웹 템플릿 (예측 결과 표시)
├── DATA/
│   ├── train/train_data.csv
│   └── test/test_data.csv
└── manage.py
```

---

## 학습 및 평가

### 데이터 전처리

* 입력: 64×64=4096 픽셀 → 0\~1 정규화
* 라벨: CSV 첫 열 (1\~111) → 0-index 변환 후 원핫 인코딩
* 정규화 통계: `norm_stats.npz`에 mean/std 저장

### 학습 파이프라인

* 명령어:

  ```
  python -m core.train_any all
  ```
* MLP & CNN 모두 학습 (기존 모델 존재 시 skip)
* Best Checkpoint 저장
* 정규화 통계 재계산

### CNN 학습 로그 예시

```
[CNN] Epoch 5/30   Loss=3.40  Acc=26.67%
[CNN] Epoch 10/30  Loss=0.59  Acc=85.32%
[CNN] Epoch 20/30  Loss=0.07  Acc=99.64%
[CNN] Test Loss=2.09, Acc=53.75%   # 초기 버전
```

➡ Data Augmentation + Dropout 추가 후: 테스트 정확도 \~70%+ 달성

---

## Django 웹 시각화

* 경로: `http://127.0.0.1:8000/visualize/`
* 동작:

  1. 무작위 샘플 1장 선택
  2. MLP와 CNN 각각 예측
  3. 실제 라벨과 비교 후 시각화

### 예시 화면

| MLP 예측              | CNN 예측                |
| ------------------- | --------------------- |
| Pred: 23 (True: 23) | Pred: 45 ❌ (True: 12) |

---

## 주요 성과

* 딥러닝 프레임워크 없이 CNN/MLP를 NumPy로 직접 구현
* 모델 학습 안정화 → 50% → 70%+ 테스트 정확도 개선
* Django 웹과 AI 모델을 통합한 엔드투엔드 파이프라인 구축 경험
* 데이터 전처리, 오버피팅 완화, 성능 개선 기법을 종합적으로 적용
* AI Engineer 면접에서 **“모델 직접 구현 + 웹 연동 + 성능 개선”** 사례로 활용 가능

---

이렇게 정리하면, \*\*기술적 깊이(모델 구현/수학적 세부사항)\*\*와 \*\*엔드투엔드 설계 경험(웹까지 포함)\*\*이 명확히 드러납니다.

원하시면 제가 README에 **MLP vs CNN 성능 비교 표**와 \*\*성능 개선 전/후 그래프 (Loss & Accuracy 곡선)\*\*도 추가해서 면접용 자료로 더 강화해드릴까요?
