import pytesseract
from pdf2image import convert_from_path
import cv2
import numpy as np
import re
from PIL import Image
from cv2 import dnn_superres

# Tesseract 실행 경로 설정
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# PDF 파일 경로
pdf_path = "/Users/william/Desktop/ChangePDF/example.pdf"

# PDF -> 이미지
images = convert_from_path(pdf_path)

# FSRCNN 모델 로드 함수
def apply_fsrcnn(img):
    sr = dnn_superres.DnnSuperResImpl_create()
    # FSRCNN 모델 경로 (.pb 파일) - 사전에 다운로드 필요
    model_path = "/Users/william/Documents/GitHub/ChangePDF/FSRCNN_x2.pb"  # 적절한 경로로 수정

    sr.readModel(model_path)
    sr.setModel("fsrcnn", 2)  # FSRCNN 모델과 업스케일 비율 (x2)
    upscaled = sr.upsample(img)
    return upscaled

#속도 향상을 위한 이미지 사이즈 조정
def resize_image(image, max_width=1200):
    h, w = image.shape[:2]
    if w > max_width:
        ratio = max_width / w
        return cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
    return image


# 전처리 함수
def preprocess_image(image):
    img = np.array(image)

    # FSRCNN 적용
    img = resize_image(img)  # FSRCNN 전에 리사이즈
    img = apply_fsrcnn(img)

    # 그레이스케일
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 가우시안 블러
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 이진화
    processed = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )

    return Image.fromarray(processed)

# OCR 함수
def extract_text(image):
    processed_image = preprocess_image(image)

    text_psm11 = pytesseract.image_to_string(processed_image, lang="kor+eng", config="--oem 3 --psm 11")
    text_psm6 = pytesseract.image_to_string(processed_image, lang="kor+eng", config="--oem 3 --psm 6")

    return text_psm11 if len(text_psm11) > len(text_psm6) else text_psm6

# 결과 저장용 리스트
code, name, unit, count, price, amount, sum_ = [], [], [], [], [], [], []

# 페이지별 OCR 처리
for i, image in enumerate(images):
    text = extract_text(image)
    print(f"Page {i+1}:\n{text}\n{'-'*50}")

    pattern = r"(\d+)\s+([^\d]+)\s+([\d\w/.%()]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d,]+)"
    matches = re.findall(pattern, text)

    for match in matches:
        code.append(match[0])
        name.append(match[1].strip())
        unit.append(match[2])
        count.append(match[3])
        price.append(match[4])
        amount.append(match[5])
        sum_.append(match[6])

# 출력
print("code:", code)
print("name:", name)
print("unit:", unit)
print("count:", count)
print("price:", price)
print("amount:", amount)
print("sum:", sum_)
print("갯수: ", len(name))
