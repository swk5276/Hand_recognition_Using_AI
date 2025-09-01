from django.db import models

from core.MLP_Neural_Network import NeuralNetwork, load_data
# 또는 새 파일명을 쓰는 중이면
from core.mlp_numpy import NeuralNetwork
from core.data import load_csv_dataset as load_data
# 이 파일에서 NeuralNetwork와 load_data를 import 가능하도록 설정
__all__ = ['NeuralNetwork', 'load_data']
