import pandas as pd
import os

# 최상위 폴더 경로 설정
train_folder_path = "C:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\train"

# CSV 파일을 저장할 리스트 초기화
csv_files = []

# train 폴더 내 모든 하위 폴더와 파일 순회
for root, dirs, files in os.walk(train_folder_path):
    for file in files:
        if file.endswith(".csv"):
            # CSV 파일의 전체 경로를 리스트에 추가
            csv_files.append(os.path.join(root, file))

# CSV 파일이 있는지 확인하고 병합
if csv_files:
    # 각 CSV 파일을 데이터프레임으로 읽어와서 병합
    merged_data = pd.concat([pd.read_csv(file, header=None) for file in csv_files], ignore_index=True)

    # 병합된 데이터프레임을 하나의 CSV 파일로 저장
    merged_csv_file_path = os.path.join(train_folder_path, "train3_data.csv")
    merged_data.to_csv(merged_csv_file_path, index=False, header=False)

    print(f"모든 CSV 파일이 성공적으로 병합되어 {merged_csv_file_path}에 저장되었습니다.")






