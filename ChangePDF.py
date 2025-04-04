# import pytesseract
# from pdf2image import convert_from_path
# import cv2
# import numpy as np
# import re
# from PIL import Image

# # Tesseract 실행 경로 설정
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# # PDF 파일 경로 설정
# pdf_path = "/Users/william/Desktop/ChangePDF/example.pdf"

# # PDF를 이미지로 변환
# images = convert_from_path(pdf_path)

# # 결과 저장할 리스트
# code, name, unit, count, price, amount, sum_ = [], [], [], [], [], [], []

# # 이미지 전처리 함수 (OpenCV 활용)
# def preprocess_image(image):
#     img = np.array(image)  # PIL 이미지를 NumPy 배열로 변환

#     # 그레이스케일 변환
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # 가우시안 블러 적용 (노이즈 제거)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 적응형 이진화 (대비 증가)
#     processed = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

#     return Image.fromarray(processed)

# # 두 개의 OCR 방식으로 비교하는 함수
# def extract_text(image):
#     processed_image = preprocess_image(image)  # 전처리 적용

#     # psm 11: 흐릿한 텍스트도 인식 가능
#     text_psm11 = pytesseract.image_to_string(processed_image, lang="kor+eng", config="--oem 3 --psm 11")

#     # psm 6: 표 형식 데이터에 적합
#     text_psm6 = pytesseract.image_to_string(processed_image, lang="kor+eng", config="--oem 3 --psm 6")

#     # 두 결과 중 더 긴 쪽 선택 (일반적으로 더 정확함)
#     return text_psm11 if len(text_psm11) > len(text_psm6) else text_psm6

# # OCR을 수행하여 텍스트 추출 및 데이터 정리
# for i, image in enumerate(images):
#     text = extract_text(image)  # 최적의 OCR 결과 사용
#     print(f"Page {i+1}:\n{text}\n{'-'*50}")  # OCR 결과 출력 (디버깅용)
    
#     # 정규표현식을 이용한 데이터 추출 패턴
#     pattern = r"(\d+)\s+([^\d]+)\s+([\d\w/.%()]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)"

#     # 데이터 추출
#     matches = re.findall(pattern, text)

#     for match in matches:
#         code.append(match[0])
#         name.append(match[1].strip())  # 공백 제거
#         unit.append(match[2])
#         count.append(match[3])
#         price.append(match[4])
#         amount.append(match[5])
#         sum_.append(match[6])

# # 결과 출력
# print("code:", code)
# print("name:", name)
# print("unit:", unit)
# print("count:", count)
# print("price:", price)
# print("amount:", amount)
# print("sum:", sum_)
# print("갯수: ", len(name))


###### Easy OCR을 활용한 코드
# import easyocr
# from pdf2image import convert_from_path
# import cv2
# import numpy as np
# import re
# from PIL import Image

# # EasyOCR 리더 초기화 (한글+영어 지원)
# reader = easyocr.Reader(['ko', 'en'])

# # PDF 파일 경로 설정
# pdf_path = "/Users/william/Desktop/ChangePDF/example.pdf"

# # PDF를 이미지로 변환
# images = convert_from_path(pdf_path, dpi=300)

# # 결과 저장할 리스트
# code, name, unit, count, price, amount, sum_ = [], [], [], [], [], [], []

# # 이미지 전처리 함수 (OpenCV 활용)
# def preprocess_image(image):
#     img = np.array(image)  # PIL 이미지를 NumPy 배열로 변환

#     # 그레이스케일 변환
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # 가우시안 블러 적용 (노이즈 제거)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)

#     # 적응형 이진화 (대비 증가)
#     processed = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

#     return Image.fromarray(processed)

# # EasyOCR을 이용한 텍스트 추출 함수
# def extract_text(image):
#     processed_image = preprocess_image(image)  # 전처리 적용

#     # EasyOCR 실행
#     results = reader.readtext(np.array(processed_image), detail=0)  # detail=0은 텍스트만 반환
#     text = "\n".join(results)  # 줄 단위로 텍스트 결합

#     return text

# # OCR을 수행하여 텍스트 추출 및 데이터 정리
# for i, image in enumerate(images):
#     text = extract_text(image)  # EasyOCR 결과 사용
#     print(f"Page {i+1}:\n{text}\n{'-'*50}")  # OCR 결과 출력 (디버깅용)
    
#     # 정규표현식을 이용한 데이터 추출 패턴
#     pattern = r"(\d+)\s+([^\d]+)\s+([\d\w/.%()]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)"

#     # 데이터 추출
#     matches = re.findall(pattern, text)

#     for match in matches:
#         code.append(match[0])
#         name.append(match[1].strip())  # 공백 제거
#         unit.append(match[2])
#         count.append(match[3])
#         price.append(match[4])
#         amount.append(match[5])
#         sum_.append(match[6])

# # 결과 출력
# print("code:", code)
# print("name:", name)
# print("unit:", unit)
# print("count:", count)
# print("price:", price)
# print("amount:", amount)
# print("sum:", sum_)
# print("갯수: ", len(name))



############ 정확도를 높이기 위해 EasyOCR 튜닝 & EasyOCR과 Tytesseract 동시에 사용.
import easyocr
import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
from PIL import Image

# Tesseract 실행 경로 설정 (Mac 사용자용)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# EasyOCR 리더 초기화 (한글+영어 지원)
reader = easyocr.Reader(['ko', 'en'])

# PDF 파일 경로 설정
pdf_path = "/Users/william/Desktop/ChangePDF/example.pdf"

# PDF를 이미지로 변환 (dpi=300 설정)
images = convert_from_path(pdf_path, dpi=300)

# 결과 저장할 리스트
code, name, unit, count, price, amount, sum_ = [], [], [], [], [], [], []

# 이미지 전처리 함수 (OpenCV 활용)
def preprocess_image(image):
    img = np.array(image)  # PIL 이미지를 NumPy 배열로 변환

    # 그레이스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 가우시안 블러 적용 (노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 적응형 이진화 (대비 증가)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # 이미지 확대 (해상도 증가)
    processed = cv2.resize(binary, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(processed)

# OCR을 이용한 텍스트 추출 함수 (EasyOCR + Tesseract)
def extract_text(image):
    processed_image = preprocess_image(image)  # 전처리 적용

    # EasyOCR 실행 (최적화 설정 적용)
    easy_text = reader.readtext(np.array(processed_image), detail=0, 
                                contrast_ths=0.3, adjust_contrast=0.7, decoder='beamsearch')
    easy_text = "\n".join(easy_text)  # 줄 단위로 텍스트 결합

    # Tesseract 실행 (표 데이터에 적합한 psm 6 설정)
    tess_text = pytesseract.image_to_string(processed_image, lang="kor+eng", config="--oem 3 --psm 6")

    # 두 OCR 결과 중 더 신뢰할 수 있는 결과 선택
    final_text = easy_text if len(easy_text) > len(tess_text) else tess_text
    return final_text

# OCR을 수행하여 텍스트 추출 및 데이터 정리
for i, image in enumerate(images):
    text = extract_text(image)  # 최적의 OCR 결과 사용
    print(f"Page {i+1}:\n{text}\n{'-'*50}")  # OCR 결과 출력 (디버깅용)
    
    # 정규표현식을 이용한 데이터 추출 패턴
    pattern = r"(\d+)\s+([^\d]+)\s+([\d\w/.%()]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)"

    # 데이터 추출
    matches = re.findall(pattern, text)

    for match in matches:
        code.append(match[0])
        name.append(match[1].strip())  # 공백 제거
        unit.append(match[2])
        count.append(match[3])
        price.append(match[4])
        amount.append(match[5])
        sum_.append(match[6])

# 결과 출력
print("code:", code)
print("name:", name)
print("unit:", unit)
print("count:", count)
print("price:", price)
print("amount:", amount)
print("sum:", sum_)
print("갯수: ", len(name))




###Google Cloud Vision OCR 방식도 있는데 이건.. 유료...
