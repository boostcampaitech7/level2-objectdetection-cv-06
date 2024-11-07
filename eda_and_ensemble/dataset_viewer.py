import streamlit as st
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import os

# 클래스 이름 정의
CLASS_NAMES = ['General trash', 'Paper', 'Paper pack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing']

# 클래스별 색상 정의 (RGB 형식)
CLASS_COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),    # Maroon
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Navy
    (128, 128, 0)   # Olive
]

# CSV 파일 로드
@st.cache_data
def load_csv(file_name):
    csv_path = os.path.join(os.path.dirname(__file__), 'csv', file_name)
    df = pd.read_csv(csv_path)
    df['image_id'] = df['image_id'].apply(lambda x: x.split('/')[-1])  # 파일명만 추출
    df = df.sort_values('image_id')  # 이미지 ID로 정렬
    return df

# 이미지 로드
@st.cache_data
def load_image(image_path):
    return Image.open(image_path)

# 바운딩 박스 그리기
def draw_bbox(image, bbox, label, score, color):
    draw = ImageDraw.Draw(image, 'RGBA')
    
    # 박스 그리기
    draw.rectangle(bbox, outline=color, width=3)
    
    # 폰트 설정
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 100)
    except IOError:
        font = ImageFont.load_default()
    
    label = f"{label}: {score:.2f}"
    
    # 텍스트 크기 계산
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # 텍스트 위치 설정 (박스 위에 배치)
    x = bbox[0]
    y = bbox[1] - text_height - 5
    
    # 반투명 배경 그리기
    bg_color = color + (128,)  # alpha value for semi-transparency
    draw.rectangle([x, y, x + text_width, y + text_height], fill=bg_color)
    
    # 텍스트에 외곽선 추가
    for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
        draw.text((x+offset[0], y+offset[1]), label, font=font, fill=(0, 0, 0, 255))
    
    # 메인 텍스트 그리기
    draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
    
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
    test_image_dir = "../dataset/test"

    if os.path.exists(test_image_dir):
        image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        if 'image_index' not in st.session_state:
            st.session_state.image_index = 0

        # 이미지 선택을 위한 UI 구성
        col1, col2, col3, col4 = st.columns([1, 3, 1, 2])
        with col1:
            if st.button("➖"):
                st.session_state.image_index = max(0, st.session_state.image_index - 1)
        with col2:
            st.session_state.image_index = st.slider("Select an image", 0, len(image_files)-1, st.session_state.image_index)
        with col3:
            if st.button("➕"):
                st.session_state.image_index = min(len(image_files)-1, st.session_state.image_index + 1)
        with col4:
            input_index = st.number_input("Enter image number", min_value=0, max_value=len(image_files)-1, value=st.session_state.image_index)
            if input_index != st.session_state.image_index:
                st.session_state.image_index = input_index

        selected_image = image_files[st.session_state.image_index]

        # 예측 확률 임계값 설정을 위한 슬라이더 추가
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

        if selected_image:
            image_path = os.path.join(test_image_dir, selected_image)
            image = load_image(image_path)

            st.image(image, caption=f"Original Image (Image {st.session_state.image_index})", use_column_width=True)

            # CSV 파일에서 해당 이미지의 예측 결과 가져오기
            predictions1 = df1[df1['image_id'] == selected_image]
            predictions2 = df2[df2['image_id'] == selected_image]

            # 두 모델의 예측 결과 시각화
            st.subheader(f"{csv_file1} Predictions")
            image1 = image.copy()
            annotation_count1 = 0
            for _, row in predictions1.iterrows():
                pred_strings = row['PredictionString'].split()
                for i in range(0, len(pred_strings), 6):
                    label = int(pred_strings[i])
                    score = float(pred_strings[i+1])
                    if score >= confidence_threshold:  # 임계값 이상의 예측만 표시
                        annotation_count1 += 1
                        bbox = list(map(float, pred_strings[i+2:i+6]))
                        image1 = draw_bbox(image1, bbox, CLASS_NAMES[label], score, CLASS_COLORS[label])
            st.image(image1, caption=f"{csv_file1} Predictions", use_column_width=True)
            st.write(f"Number of annotations: {annotation_count1}")

            st.subheader(f"{csv_file2} Predictions")
            image2 = image.copy()
            annotation_count2 = 0
            for _, row in predictions2.iterrows():
                pred_strings = row['PredictionString'].split()
                for i in range(0, len(pred_strings), 6):
                    label = int(pred_strings[i])
                    score = float(pred_strings[i+1])
                    if score >= confidence_threshold:  # 임계값 이상의 예측만 표시
                        annotation_count2 += 1
                        bbox = list(map(float, pred_strings[i+2:i+6]))
                        image2 = draw_bbox(image2, bbox, CLASS_NAMES[label], score, CLASS_COLORS[label])
            st.image(image2, caption=f"{csv_file2} Predictions", use_column_width=True)
            st.write(f"Number of annotations: {annotation_count2}")

        else:
            st.error("Test image directory not found.")

if __name__ == "__main__":
    main()