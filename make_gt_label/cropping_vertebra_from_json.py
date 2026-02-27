import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

import sys
import os

# 1. 현재 파일의 부모 폴더(분당서울대병원) 위치를 파이썬 뇌에 강제 주입!
current_dir = os.path.dirname(os.path.abspath(__file__)) # 현재: make_gt_label
parent_dir = os.path.dirname(current_dir)                # 부모: snubh
sys.path.append(parent_dir)
from utils import makelabelmeFile, preprocessImage, keypointDetection

# ==========================================
# 1. 설정
# 환자 단위로 train / validataion split
# Acute를 더 우선적으로 순위를 둠
# ==========================================

POS = 'lat'  # 'AP' or 'lat' (파일 찾기 및 저장명에 사용)
MODE = 'train'
prep = preprocessImage()

# 경로 설정
ROOT_FOLDER = Path()

# JSON 파일들이 들어있는 폴더 (보통 이미지와 같이 있음)
# INPUT_FOLDER = ROOT_FOLDER / "Crescom_251223_fracture" / MODE 
# INPUT_FOLDER = Path("/home/user/Desktop/251217_compression/test_data_json_0203")
INPUT_FOLDER = Path(f"Converting_from_dcm_{MODE}")

# 결과 저장 폴더
OUTPUT_FOLDER = Path(f"cropping_vertebra_result_from_json_{POS}_{MODE}")

# Crop 여백 설정 (기존 코드의 margin_ratio와 유사)
PADDING_RATIO = 0.08  # 박스 크기의 8% 만큼 상하좌우 여백 추가

# 결과 이미지 확장자
SAVE_EXT = ".jpg"

if MODE == 'train':
    VAL_RATIO = 0.2
else:
    VAL_RATIO = 0
SEED = 42


# ==========================================
# 2. 실행 함수
# ==========================================

def crop_from_json(input_folder, output_folder, position_str):
    output_folder.mkdir(parents=True, exist_ok=True)
    
    summary_results = []
    
    # JSON 파일 찾기
    print(f"Searching for JSON files in {input_folder}...")
    json_files = list(input_folder.rglob("*.json"))
    
    # position_str(lat/ap)이 파일명이나 경로에 포함된 것만 필터링 (필요하다면)
    # json_files = [p for p in json_files if position_str.lower() in p.name.lower()]
    
    print(f"Found {len(json_files)} JSON files.")
    
    for json_path in tqdm(json_files):
        try:
            # 1. JSON 로드
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # file_name = json_path.stem
            # reg_id_raw = file_name.split("_")[0]
            # reg_id = str(int(reg_id_raw))
            
            # 2. 이미지 경로 찾기 (JSON과 같은 폴더에 있다고 가정)
            image_name = data.get("imagePath", "")
            # imagePath가 비어있으면 JSON 파일명으로 유추
            if not image_name:
                image_name = json_path.stem + ".jpg" # 기본 jpg 가정

            image_path = json_path.parent / image_name
            
            # 확장자가 다를 수 있으므로 파일 존재 확인
            if not image_path.exists():
                possible_exts = ['.jpg', '.png', '.dcm', '.dicom', '.JPG', '.PNG']
                found = False
                for ext in possible_exts:
                    temp_path = json_path.parent / (json_path.stem + ext)
                    if temp_path.exists():
                        image_path = temp_path
                        found = True
                        break
                if not found:
                    # print(f"[SKIP] Image not found for: {json_path.name}")
                    continue

            # 3. 이미지 로드 (한글 경로 등 호환성을 위해 imdecode 사용)
            # img_array = np.fromfile(str(image_path), np.uint8)
            # image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # 이미지 로드
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                image = prep.convert_dicom2img(image_path)
            else:
                image = cv2.imread(str(image_path))
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # RegID 추출 (폴더명 예: '0001_AP_Lat' -> '1')
            try:
                # folder_name = json_path.parent.name
                # 폴더명 규칙에 따라 수정 필요. 보통 맨 앞 숫자가 ID
                reg_id_raw = json_path.stem.split('_')[0] 
                reg_id = str(int(reg_id_raw))
            except:
                # reg_id = json_path.parent.name # 실패시 폴더명 그대로 사용
                reg_id = json_path.stem

            # 4. Shapes(Label) 순회 및 Crop
            for shape in data.get("shapes", []):
                label_raw = shape.get("label", "Unknown")
                points = np.array(shape.get("points", []))
                
                if len(points) == 0:
                    continue

                # ------------------------------------------------
                # 라벨 파싱 (예: L1_Acute -> Bone: L1, Diagnosis: Acute)
                # ------------------------------------------------
                if '_' in label_raw:
                    parts = label_raw.split('_')
                    bone_name = parts[0].upper()
                    diagnosis = parts[1].capitalize() # Acute, Normal 등
                    
                    if diagnosis.lower() == 'vp':
                        diagnosis = "VP"
                    
                    # 만약 L1_Acute_Wedge 처럼 뒤에 더 붙는 경우 처리
                    if len(parts) > 2:
                        diagnosis = "_".join(parts[1:])
                else:
                    # 언더바가 없는 경우 (그냥 Acute 등) -> 뼈 이름은 Unknown 혹은 라벨 그대로
                    bone_name = "Unknown" 
                    diagnosis = label_raw.capitalize()
                    
                    # 혹시 라벨 자체가 뼈 이름일 수도 있음 (상황에 맞춰 수정)
                    # bone_name = label_raw.upper()
                    # diagnosis = "Unknown"

                # ------------------------------------------------
                # Crop 좌표 계산 (Padding 적용)
                # ------------------------------------------------
                x_vals = points[:, 0]
                y_vals = points[:, 1]
                
                x_min_raw = min(x_vals)
                x_max_raw = max(x_vals)
                y_min_raw = min(y_vals)
                y_max_raw = max(y_vals)
                
                vb_w = x_max_raw - x_min_raw
                vb_h = y_max_raw - y_min_raw
                
                # 여백 계산
                margin_x = int(vb_w * PADDING_RATIO)
                margin_y = int(vb_h * PADDING_RATIO)
                
                x_min = int(max(0, x_min_raw - margin_x))
                y_min = int(max(0, y_min_raw - margin_y))
                x_max = int(min(w, x_max_raw + margin_x))
                y_max = int(min(h, y_max_raw + margin_y))
                
                # 너무 작은 박스는 스킵
                if (x_max - x_min) < 5 or (y_max - y_min) < 5:
                    continue
                
                # 자르기
                cropped_img = image[y_min:y_max, x_min:x_max]
                
                if cropped_img.size == 0:
                    continue

                # ------------------------------------------------
                # 저장 (Diagnosis 별 폴더 구분)
                # ------------------------------------------------
                # 예: Output/Acute/
                class_dir = output_folder / diagnosis
                class_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일명: ID_Pos_Bone_Diag.jpg (예: 1_LAT_L1_Acute.jpg)
                save_name = f"{reg_id}_{position_str.upper()}_{bone_name}_{diagnosis}{SAVE_EXT}"
                save_path = class_dir / save_name
                
                # 이미지 저장 
                extension = os.path.splitext(save_name)[1]
                # result, encoded_img = cv2.imencode(extension, cropped_img)
                # if result:
                #     with open(str(save_path), mode='w+b') as f:
                #         encoded_img.tofile(f)
                # result, encoded_img = cv2.imwrite(extension, cropped_img)
                cv2.imwrite(str(save_path), cropped_img)
                
                # 요약 리스트 추가
                summary_results.append({
                    "RegID": reg_id,
                    "Vertebra": bone_name,
                    "Position": position_str.upper(),
                    "Diagnosis": diagnosis,
                    "Original_File": json_path.stem,
                    "Crop_Image_Path": str(save_path.resolve()),
                    "Label_Raw": label_raw
                })
                
        except Exception as e:
            print(f"[ERROR] {json_path.name}: {e}")
            continue

    # ------------------------------------------------
    # 엑셀 저장
    # ------------------------------------------------
    if summary_results:
        df = pd.DataFrame(summary_results)
        excel_path = output_folder / f"crop_summary_from_json_{position_str}.xlsx"
        df.to_excel(excel_path, index=False)
        
        if MODE == 'train':
            patient_diagnosis_map = df.groupby('RegID')['Diagnosis'].apply(list).to_dict()
            unique_ids = list(patient_diagnosis_map.keys())
            
            stratify_labels = []
            
            for uid in unique_ids:
                diags = patient_diagnosis_map[uid]
                
                # Acute, VP 있으면 Acute 먼저
                if 'Acute' in diags:
                    final_label = 'Acute'
                elif 'Chronic' in diags:
                    final_label = 'Chronic'
                elif 'VP' in diags:
                    final_label = 'VP'
                else:
                    final_label = 'Normal'
                    
                stratify_labels.append(final_label)
                
            try:
                train_ids, val_ids = train_test_split(
                    unique_ids, 
                    test_size=VAL_RATIO, 
                    random_state=SEED, 
                    stratify=stratify_labels
                )
                
            except ValueError as e:
                print(f"[WARN] Stratify 실패 (클래스 불균형 심함): {e}")
                print("랜덤 Split으로 진행합니다.")
                train_ids, val_ids = train_test_split(
                    unique_ids, 
                    test_size=VAL_RATIO, 
                    random_state=SEED
                )
            
            print(f"Train Patients: {len(train_ids)}")
            print(f"Val   Patients: {len(val_ids)}")
            
            train_df = df[df['RegID'].isin(train_ids)]
            val_df = df[df['RegID'].isin(val_ids)]
            
            print(f"Train Image Distribution:\n{train_df['Diagnosis'].value_counts()}")
            print(f"Val Image Distribution:\n{val_df['Diagnosis'].value_counts()}")
            
            train_save_path = output_folder / "train_list.csv"
            val_save_path = output_folder / "val_list.csv"
            
            train_df.to_csv(train_save_path, index=False)
            val_df.to_csv(val_save_path, index=False) # Val은 건드리지 않음!
        else:
            save_path = output_folder / "test_list.csv"
            df.to_csv(save_path, index=False)
            
        print("\n" + "="*50)
        print(f"[완료] 데이터셋 생성 끝!")
        print(f"저장 위치: {output_folder}")
        print(f"총 이미지 수: {len(df)}")
        print(f"클래스별 개수:\n{df['Diagnosis'].value_counts()}")
        print("="*50)
    else:
        print("[WARN] 결과가 없습니다.")

# ==========================================
# 3. Main 실행
# ==========================================
if __name__ == "__main__":
    crop_from_json(INPUT_FOLDER, OUTPUT_FOLDER, POS)