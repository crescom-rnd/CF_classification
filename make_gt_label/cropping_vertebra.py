import os
import cv2
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from utils import makelabelmeFile, preprocessImage, keypointDetection

# from module1_re import Module1

##### 척추체 단위 이미지 만드는 코드 #####
##### 압박 골절 labeling csv(GT) + module1 으로 척추체 찾아서 전체 척추체에 대한 xlsx 만듬 #####
##### file 명 & file 위치 필요 #####


POS = 'lat'  # <--- 'AP' or 'lat'
DEVICE_NUM = 0
MODE = "test"

MLM = makelabelmeFile()

# 경로 설정
ROOT_FOLDER = Path("/home/user/Desktop/251217_compression")
INPUT_FOLDER = ROOT_FOLDER / "Crescom_251223_fracture" / MODE
# EXCEL_PATH = Path(f"make_gt_label/gt_{MODE}.xlsx")  # load할 엑셀 파일 경로 확인
EXCEL_PATH = Path(f"original_test.xlsx")  # load할 엑셀 파일 경로 확인
OUTPUT_FOLDER = Path(f"cropping_vertebra_result_0127_{POS}_{MODE}_temp_edit")

VAL_RATIO = 0.2
SEED = 42

class Module1:
    def __init__(self, device_num=0, position='LAT'):
        self.names = [
            's1-1', 's1-2', 'l5-3', 'l5-4', 'l5-1', 'l5-2', 'l4-3', 'l4-4', 'l4-1', 'l4-2',
            'l3-3', 'l3-4', 'l3-1', 'l3-2', 'l2-3', 'l2-4', 'l2-1', 'l2-2', 'l1-3', 'l1-4', 'l1-1', 'l1-2',
            't12-3', 't12-4', 't12-1', 't12-2', 't11-3', 't11-4', 't11-1', 't11-2', 't10-3', 't10-4', 't10-1', 't10-2',
            't9-3', 't9-4', 't9-1', 't9-2', 't8-3', 't8-4', 't8-1', 't8-2', 't7-3', 't7-4', 't7-1', 't7-2',
            't6-3', 't6-4', 't6-1', 't6-2', 't5-3', 't5-4', 't5-1', 't5-2'
        ]
        self.position = position.upper()
        # 가중치 경로가 position에 따라 달라짐 (사용자 환경에 맞게 경로 확인)
        w_dir = f"weight/{self.position}"  # 예: weight/LAT/
        c_dir = f"config/{self.position}"
        
        self.keymodels = [
            [device_num, 'pose_hrnet', f'{w_dir}/1st_1.pth', 12, 384, 288, ['0','1','2','3','4','5','6','7','8','9','10','11'], [f'{c_dir}/1st_1.yaml', 'MODEL', 'EXTRA']],
            [device_num, 'pose_hrnet', f'{w_dir}/1st_2.pth', 12, 384, 288, ['0','1','2','3','4','5','6','7','8','9','10','11'], [f'{c_dir}/1st_2.yaml', 'MODEL', 'EXTRA']],
            [device_num, 'pose_hrnet', f'{w_dir}/2nd.pth', 12, 384, 288, ['0','1','2','3','4','5','6','7','8','9','10','11'], [f'{c_dir}/2nd.yaml', 'MODEL', 'EXTRA']],
            # [device_num, 'pose_hrnet', f'{w_dir}/3rd_6.pth', 6, 384, 288, ['0','1','2','3','4','5'], [f'{c_dir}/3rd_6.yaml', 'MODEL', 'EXTRA']]
        ]
        
        # 파라미터 설정
        if self.position == 'LAT':
            self.max_y_delta = 1
            self.top_dist = 8
            self.bot_dist = 6
        else: # AP
            self.max_y_delta = 1/3
            self.top_dist = 6
            self.bot_dist = 5
            
        self.KEY = keypointDetection()
        self.KEY.keypoint_load_model(self.keymodels)

    def inference_4point(self, image):
        final_coords = []
        ori_h, w = image.shape[:2]
        
        # 1. First Inference (Whole Image)
        image, coords, labels, scores = self.KEY.keypoint_detect2(image, 0, [0, 0, w, ori_h], 288, 384, 0.1)

        x_distances = []
        for coord_set in zip(coords[::2], coords[1::2]):
            if coord_set[0][0] is not None and coord_set[0][1] is not None and coord_set[1][0] is not None and coord_set[1][1] is not None:
                x_distances.append(distance.euclidean(coord_set[0], coord_set[1]))

        valid_coords = [c for c in coords if c is not None and c[0] is not None and c[1] is not None]
        if not valid_coords:
            pass
            # 감지 실패 시 None 반환 혹은 예외 처리
            # return image, dict(zip(self.names, [None]*len(self.names)))

        distance_mean = np.mean(x_distances) if x_distances else 0
        _, y_mean = np.mean(valid_coords, axis=0)
        h = ori_h

        # 2. ROI Refinement (높이 기반 자르기)
        if distance_mean > 0 and (y_mean > distance_mean * self.top_dist or (ori_h - y_mean) > distance_mean * self.bot_dist):
            y_min = max(0, y_mean - distance_mean * self.top_dist)
            y_max = min(ori_h, y_mean + distance_mean * self.bot_dist)
            h = y_max - y_min
            image, coords, labels, scores = self.KEY.keypoint_detect2(image, 0, [0, y_min, w, h], 288, 384, 0.1)
        else:
            h = ori_h

        # 3. Validation & Re-inference logic
        for coord in coords:
            if coord[1] is None or coord[1] > ori_h:
                image, coords, labels, scores = self.KEY.keypoint_detect2(image, 1, [0, 0, w, ori_h], 288, 384, 0.1)
                break

        # first inference
        x_distances = []
        for coord_set in zip(coords[::2], coords[1::2]):
            if coord_set[0][0] is not None and coord_set[0][1] is not None and coord_set[1][0] is not None and coord_set[1][1] is not None:
                x_distances.append(distance.euclidean(coord_set[0], coord_set[1]))

        if len(x_distances) == 0 \
        or coords[0][0] is None or coords[0][1] is None or coords[1][0] is None or coords[1][1] is None\
        or coords[0][1] < 0 or coords[1][0] < 0:
            next_flag = False

        else:
            x_distance_mean = np.mean(x_distances)

            min_y = min(coords[0][1], coords[1][1]) - 3 * x_distance_mean
            max_y = max(coords[0][1], coords[1][1]) + self.max_y_delta * x_distance_mean
            min_x = min(coords[0][0], coords[1][0]) - 2 * x_distance_mean
            max_x = max(coords[0][0], coords[1][0]) + 2 * x_distance_mean

            box = [min_x, min_y, max_x - min_x, max_y - min_y]

            image, coords, labels, scores = self.KEY.keypoint_detect2(image, 2, box, 288, 384, 0.1, mask_image=True)

            next_flag = True
            for coord in coords:
                x = coord[0]
                y = coord[1]
                if x is None or y is None or x < 20 or y < 20:
                    final_coords.append(None)
                    next_flag = False
                    continue
                else:
                    final_coords.append(coord)

        # 4. Iterative Inference
        for _ in range(3):
            if next_flag:
                split = np.mean([coords[8], coords[9]], axis=0)
                image, coords, labels, scores = self.KEY.keypoint_detect2(image, 1, [0, -h + split[1], w, h], 288, 384, 0.1, mask_image=True)

                x_distances = []
                for coord_set in zip(coords[::2], coords[1::2]):
                    if coord_set[0][0] is not None and coord_set[0][1] is not None and coord_set[1][0] is not None and coord_set[1][1] is not None:
                        x_distances.append(distance.euclidean(coord_set[0], coord_set[1]))

                if len(x_distances) == 0 \
                or coords[0][0] is None or coords[0][1] is None or coords[1][0] is None or coords[1][1] is None\
                or coords[0][1] < 0 or coords[1][0] < 0:
                    next_flag = False
                
                else:
                    x_distance_mean = np.mean(x_distances)
                    min_y = min(coords[0][1], coords[1][1]) - 3 * x_distance_mean
                    max_y = max(coords[0][1], coords[1][1]) + self.max_y_delta * x_distance_mean
                    min_x = min(coords[0][0], coords[1][0]) - 2 * x_distance_mean
                    max_x = max(coords[0][0], coords[1][0]) + 2 * x_distance_mean

                    box = [min_x, min_y, max_x - min_x, max_y - min_y]

                    image, coords, labels, scores = self.KEY.keypoint_detect2(image, 2, box, 288, 384, 0.1, mask_image=True)
                    
                    next_flag = True
                    for coord in coords:
                        if coord is None or coord[0] is None or coord[1] is None or coord[0] < 20 or coord[1] < 20:
                            final_coords.append(None)
                            next_flag = False
                            continue
                        else:
                            final_coords.append(coord)

        return image, dict(zip(self.names, final_coords))

# -----------------------------------------------------------------------------
# 3. Helper Functions
# -----------------------------------------------------------------------------
def load_gt_data(excel_path):
    df = pd.read_excel(excel_path)
    label_map = {}
    for _, row in df.iterrows():
        try:
            reg_id = str(int(row['RegID'])) # 001 -> '1'로 변환
        except:
            reg_id = str(row['RegID'])
        
        # vb = str(row['vb']).strip().upper()
        # stage = str(row['stage']).strip()
        vb = str(row['Vertebra']).strip().upper()
        stage = str(row['Diagnosis']).strip()
        label_map[(reg_id, vb)] = stage
    return label_map


def inference(image_files, out_folder, position_str):
    # 1. 모델 로드 (여기서 한 번만 로드)
    print(f"Loading Model for {position_str}...")
    model = Module1(device_num=DEVICE_NUM, position=position_str)
    prep = preprocessImage()
    
    # 2. GT 맵 로드
    gt_map = load_gt_data(EXCEL_PATH)
    
    # 3. 폴더 생성
    classes = ['Normal', 'Acute', 'Chronic', 'VP']
    for cls in classes:
        (out_folder / cls).mkdir(parents=True, exist_ok=True)
    # (out_folder / "Etc").mkdir(parents=True, exist_ok=True)

    # 엑셀 저장 리스트
    summary_results = []

    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files):
        try:
            # RegID 추출 (폴더명이 '0001_AP' 형식이라고 가정)
            # parent.name -> '0001_AP', split('_')[0] -> '0001', int->str -> '1'
            folder_name = image_path.parent.name
            reg_id_raw = folder_name.split('_')[0]
            reg_id = str(int(reg_id_raw))
            
            # 이미지 로드
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                image = prep.convert_dicom2img(image_path)
            else:
                image = cv2.imread(str(image_path))
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            if image is None: continue
            image_copy = image.copy()
            h, w = image.shape[:2]

            # 추론
            _, keypoints = model.inference_4point(image_copy)
            
            # 그룹화
            groups = defaultdict(list)
            for key, coord in keypoints.items():
                if coord is None or coord[0] is None: continue
                v_name = key.split('-')[0]
                groups[v_name].append(coord)
            
            # 라벨링 및 저장
            for v_name, points in groups.items():
                if len(points) < 4: continue
                
                v_name_upper = v_name.upper()
                
                # 라벨 확인
                if (reg_id, v_name_upper) in gt_map:
                    diagnosis = gt_map[(reg_id, v_name_upper)]
                else:
                    diagnosis = 'Normal'
                
                # 저장 경로
                save_dir = out_folder / (diagnosis if diagnosis in classes else "Etc")
                # save_dir = out_folder / diagnosis
                
                # Crop
                # x_vals = [p[0] for p in points]
                # y_vals = [p[1] for p in points]
                # pad = 10
                # x_min = max(0, int(min(x_vals)) - pad)
                # y_min = max(0, int(min(y_vals)) - pad)
                # x_max = min(w, int(max(x_vals)) + pad)
                # y_max = min(h, int(max(y_vals)) + pad)
                
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                
                vb_w = max(x_vals) - min(x_vals)
                vb_h = max(y_vals) - min(y_vals)
                
                margin_ratio = 0.08
                margin_x = int(vb_w * margin_ratio)
                margin_y = int(vb_h * margin_ratio)
                
                x_min = max(0, int(min(x_vals)) - margin_x)
                y_min = max(0, int(min(y_vals)) - margin_y)
                x_max = min(w, int(max(x_vals)) + margin_x)
                y_max = min(h, int(max(y_vals)) + margin_y)
                
                crop = image_copy[y_min:y_max, x_min:x_max]
                
                if crop.size > 0:
                    # 파일명: ID_Position_Vertebra_Label.jpg
                    fname = f"{reg_id}_{position_str}_{v_name_upper}_{diagnosis}.jpg"
                    save_path = save_dir / fname
                    cv2.imwrite(str(save_dir / fname), crop)
                    
                    summary_results.append({
                        "RegID": reg_id,
                        "Vertebra": v_name_upper,
                        "Position": position_str,
                        "Diagnosis": diagnosis,
                        "Original_Image_Path": str(image_path.resolve()), # 절대 경로로 변환하여 저장
                        "Crop_Image_Path": str(save_path.resolve())       # 절대 경로로 변환하여 저장
                    })


        except Exception as e:
            print(f"Error: {image_path.name} - {e}")

    if summary_results:
        print("Saving summary Excel file...")
        df_results = pd.DataFrame(summary_results)
        # 결과 폴더 안에 'crop_summary.xlsx' 이름으로 저장
        excel_save_path = out_folder / f"cropping_vertebra_summary_{position_str}.xlsx"
        df_results.to_excel(excel_save_path, index=False)
        print(f"Summary saved to {excel_save_path}")
        
        if MODE == 'train':
            train_df, val_df = train_test_split(
                df_results, 
                test_size=VAL_RATIO, 
                random_state=SEED, 
                stratify=df_results['Diagnosis']
            )
            
            train_save_path = out_folder / "train_list.csv"
            val_save_path = out_folder / "val_list.csv"
            
            train_df.to_csv(train_save_path, index=False)
            val_df.to_csv(val_save_path, index=False) # Val은 건드리지 않음!
        
    else:
        print("No results to save.")
        
# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 출력 폴더 생성
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # 이미지 리스트 수집 (User's rglob style)
    print(f"Searching for {POS} images in {INPUT_FOLDER}...")
    image_list = []
    
    # 0001_AP/* 폴더 안의 모든 파일 검색
    for p in INPUT_FOLDER.rglob(f"*_{POS}/*"):
        if p.suffix.lower() in [".dcm", ".dicom", ".jpg", ".png"]:
            image_list.append(p)
            
    if not image_list:
        print("No images found! Check path or POS variable.")
    else:
        # 실행
        inference(sorted(image_list), OUTPUT_FOLDER, POS.upper())
        print(f"Done! Check folder: {OUTPUT_FOLDER}")