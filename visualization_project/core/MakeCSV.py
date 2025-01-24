# from PIL import Image
# import pandas as pd
# import os

# # BMP 파일이 저장된 최상위 폴더 경로 설정 (train 폴더 or test 폴더)
# train_folder_path = "C:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\test"
# merge_data = []

# # train, test 폴더 내 하위 폴더를 재귀적으로 탐색
# for root, dirs, files in os.walk(train_folder_path):
#     # BMP 파일 중 숫자로 된 이름만 정렬하여 가져오기
#     bmp_files = sorted(
#         [f for f in files if f.endswith(".bmp") and os.path.splitext(f)[0].isdigit()],
#         key=lambda x: int(os.path.splitext(x)[0])
#     )
    
#     if bmp_files:  # 숫자로 된 BMP 파일이 있는 폴더만 처리
#         data = []
#         for filename in bmp_files:
#             # 파일명에서 레이블 추출 (확장자를 제외한 파일명)
#             label = os.path.splitext(filename)[0]

#             # BMP 파일 열기 및 흑백으로 변환
#             file_path = os.path.join(root, filename)
#             img = Image.open(file_path).convert('L')  # 흑백 모드로 변환
#             pixel_data = list(img.getdata())  # 픽셀 데이터를 리스트로 변환

#             # 파일 이름과 픽셀 데이터를 합쳐서 하나의 행으로 만들기
#             row = [label] + pixel_data  # 첫 번째 열에 파일 이름(레이블) 추가
#             data.append(row)

#         # 데이터프레임 생성 (컬럼 이름 없이 파일 이름과 픽셀 정보만 저장)
#         df = pd.DataFrame(data)

#         # 현재 폴더 이름을 CSV 파일명으로 설정
#         folder_name = os.path.basename(root)
#         csv_file_path = os.path.join(root, f"{folder_name}.csv")
#         df.to_csv(csv_file_path, index=False, header=False)  # 헤더 없이 CSV 파일 저장

#         print(f"{csv_file_path} 파일이 성공적으로 생성되었습니다.")
#         print(f"CSV 파일 저장 경로: {csv_file_path}")


#파일 삭제하기 
# import os

# # BMP 파일이 저장된 최상위 폴더 경로 설정 (train 폴더 or test 폴더)
# train_folder_path = "C:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\test"

# # train_folder_path 내 하위 폴더를 재귀적으로 탐색하며 '.csv' 파일 삭제
# for root, dirs, files in os.walk(train_folder_path):
#     for file in files:
#         # 파일 이름이 '11.bmp'로 끝나는 경우 삭제
#         if file.endswith(".csv"):
#             file_path = os.path.join(root, file)
#             os.remove(file_path)
#             print(f"{file_path} 파일이 삭제되었습니다.")


from PIL import Image
import pandas as pd
import os

# BMP 파일이 저장된 최상위 폴더 경로 설정 (train 폴더 or test 폴더)
train_folder_path = "C:\\Users\\motus\\OneDrive\\바탕 화면\\인공지능\\인공지능 과제2_글자\\한글글자데이터\\train3"
merge_data = []

# 글자에 따른 레이블 매핑 설정
label_mapping = {}  # 각 글자 폴더에 고유한 레이블 번호를 할당
current_label = 0    # 레이블 시작을 0으로 설정

# train_folder_path 내 하위 폴더를 정렬하여 순차적으로 탐색
subfolders = sorted([d for d in os.listdir(train_folder_path) if os.path.isdir(os.path.join(train_folder_path, d))])

for folder_name in subfolders:
    folder_path = os.path.join(train_folder_path, folder_name)

    # 레이블 번호 할당
    label = current_label
    label_mapping[folder_name] = label  # 폴더 이름에 해당하는 레이블 매핑 추가
    current_label += 1  # 다음 레이블 번호로 증가

    # BMP 파일 중 숫자로 된 이름만 정렬하여 가져오기
    bmp_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".bmp") and os.path.splitext(f)[0].isdigit()],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    if bmp_files:  # 숫자로 된 BMP 파일이 있는 폴더만 처리
        data = []
        for filename in bmp_files:
            # BMP 파일 열기 및 흑백으로 변환
            file_path = os.path.join(folder_path, filename)
            img = Image.open(file_path).convert('L')  # 흑백 모드로 변환
            pixel_data = list(img.getdata())  # 픽셀 데이터를 리스트로 변환

            # 레이블과 픽셀 데이터를 합쳐서 하나의 행으로 만들기
            row = [label] + pixel_data  # 첫 번째 열에 레이블 추가
            data.append(row)

        # 데이터프레임 생성 (컬럼 이름 없이 파일 이름과 픽셀 정보만 저장)
        df = pd.DataFrame(data)

        # 현재 폴더 이름을 CSV 파일명으로 설정
        csv_file_path = os.path.join(folder_path, f"{folder_name}.csv")
        df.to_csv(csv_file_path, index=False, header=False)  # 헤더 없이 CSV 파일 저장

        print(f"{csv_file_path} 파일이 성공적으로 생성되었습니다.")
        print(f"CSV 파일 저장 경로: {csv_file_path}")

# 레이블 매핑 결과 출력 (각 글자에 할당된 고유 레이블 확인)
print("레이블 매핑:", label_mapping)