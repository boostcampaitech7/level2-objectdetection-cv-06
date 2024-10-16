import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# CSV 파일 로드
@st.cache_data
def load_csv(file_name):
    csv_path = os.path.join(os.path.dirname(__file__), 'csv', file_name)
    return pd.read_csv(csv_path)

# 이미지 로드
def load_image(image_path):
    return Image.open(image_path)

# 바운딩 박스 그리기
def draw_bbox(image, bbox, label, score, color):
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=2)
    font = ImageFont.load_default()
    label = f"{label}: {score:.2f}"
    draw.text((bbox[0], bbox[1] - 10), label, font=font, fill=color)
    return image

def main():
    st.title("Object Detection Visualization")

    # CSV 파일 목록 가져오기
    csv_folder = os.path.join(os.path.dirname(__file__), 'csv')
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    if len(csv_files) < 2:
        st.error("Please make sure there are at least two CSV files in the 'csv' folder.")
        return

    # CSV 파일 선택
    csv_file1 = st.selectbox("Select first CSV file", csv_files)
    csv_file2 = st.selectbox("Select second CSV file", csv_files, index=1)

    df1 = load_csv(csv_file1)
    df2 = load_csv(csv_file2)

    # 테스트 이미지 디렉토리 설정
    test_image_dir = "../../dataset/test"  # 테스트 이미지 경로를 직접 지정

    if os.path.exists(test_image_dir):
        image_files = [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # 이미지 선택
        selected_image = st.selectbox("Select an image", image_files)

        if selected_image:
            image_path = os.path.join(test_image_dir, selected_image)
            image = load_image(image_path)

            st.image(image, caption="Original Image", use_column_width=True)

            # CSV 파일에서 해당 이미지의 예측 결과 가져오기
            predictions1 = df1[df1['image_id'] == f"test/{selected_image}"]
            predictions2 = df2[df2['image_id'] == f"test/{selected_image}"]

            # 두 모델의 예측 결과 시각화
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"{csv_file1} Predictions")
                image1 = image.copy()
                for _, row in predictions1.iterrows():
                    bbox = list(map(float, row['PredictionString'].split()[2:6]))
                    label = int(row['PredictionString'].split()[0])
                    score = float(row['PredictionString'].split()[1])
                    image1 = draw_bbox(image1, bbox, f"Class {label}", score, (255, 0, 0))
                st.image(image1, caption=f"{csv_file1} Predictions", use_column_width=True)

            with col2:
                st.subheader(f"{csv_file2} Predictions")
                image2 = image.copy()
                for _, row in predictions2.iterrows():
                    bbox = list(map(float, row['PredictionString'].split()[2:6]))
                    label = int(row['PredictionString'].split()[0])
                    score = float(row['PredictionString'].split()[1])
                    image2 = draw_bbox(image2, bbox, f"Class {label}", score, (0, 255, 0))
                st.image(image2, caption=f"{csv_file2} Predictions", use_column_width=True)

    else:
        st.error("Test image directory not found.")

if __name__ == "__main__":
    main()