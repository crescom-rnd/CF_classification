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
from module1 import Module1
import shutil

# from module1_re import Module1

##### 척추체 단위 이미지 만드는 코드 #####
##### 압박 골절 labeling csv(GT) + module1 으로 척추체 찾아서 전체 척추체에 대한 xlsx 만듬 #####
##### file 명 & file 위치 필요 #####


POS = 'lat'  # <--- 'AP' or 'lat'
DEVICE_NUM = 0
# MODE = "train"
MODE = "test"
# MOD1 = Module1(POS)

MLM = makelabelmeFile()

# 경로 설정
ROOT_FOLDER = Path("/home/user/Desktop/251217_compression")
INPUT_FOLDER = ROOT_FOLDER / "Crescom_251223_fracture" / MODE
# EXCEL_PATH = Path(f"make_gt_label/gt_{MODE}.xlsx")  # load할 엑셀 파일 경로 확인
# EXCEL_PATH = Path(f"original_train_edit(260127).xlsx")  # load할 엑셀 파일 경로 확인
EXCEL_PATH = Path(f"original_test_editing.xlsx")
OUTPUT_FOLDER = Path(f"{MODE}_data_json_0203") # ??

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
            [device_num, 'pose_hrnet', f'{w_dir}/3rd_6.pth', 6, 384, 288, ['0','1','2','3','4','5'], [f'{c_dir}/3rd_6.yaml', 'MODEL', 'EXTRA']]
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
        
    
    def dicom_convert(self, dicom_path):
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(np.float32)

        # 자동 윈도우 조정
        vmin, vmax = np.percentile(img, [1, 99])
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        img = (img * 255).astype(np.uint8)

        return img
    
    def check_flipped(self, coordinates, count=2):
        flips = 0

        for small, big in zip(coordinates[::2], coordinates[1::2]):
            if small is not None and big is not None:
                if small > big:
                    flips += 1
        if flips >= count:
            return True
        else:
            return False
        
    def flip(self, width, height, type, data, axis):
        """
        데이터에 대한 flip 을 진행 한 후 flip 후의 이미지, 키포인트를 리턴하는 함수
        image (numpy) : input image
        axis (int) : flip 방향 (1 - 좌우 반전, 0 - 상하반전)
        keypoints (list) : keypoint 도 전처리 할 경우 사용([[x0, y0], [x1, y1], ....])
        """
        centerX = width/2 ; centerY = height/2
        if type == 'image':
            flipped = cv2.flip(data, axis)


        if type == 'list':
            flipped = []
            for keypoint in data:
                if keypoint is not None:
                    if axis == 1:
                        pointX = 2*centerX -keypoint[0]
                        pointY = keypoint[1]
                    else:
                        pointX = keypoint[0]
                        pointY = 2 * centerY - keypoint[1]
                    flipped.append([pointX, pointY])
                        # cv2.circle(show_flipped, (int(pointX), int(pointY)), 20,255,-1)
        if type == 'dict':
            flipped = {}
            for key in data:
                
                if None not in data[key]:
                    keypoint = data[key]
                    if axis == 1:
                        pointX = 2*centerX -keypoint[0]
                        pointY = keypoint[1]
                    else:
                        pointX = keypoint[0]
                        pointY = 2 * centerY - keypoint[1]
                    flipped[key] = [pointX, pointY]
                else:
                    flipped[key] = [None, None]
                    # cv2.circle(show_flipped, (int(pointX), int(pointY)), 20,255,-1)
            # cv2.imshow('dd', imutils.resize(show_flipped, height=800))
            # cv2.waitKey(0)
        return flipped
    
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
    
    
    def inference_6point(self, image):
        """
        return: 6-points coordinates
        """
        image_result, keypoints_dict4 = self.inference_4point(image)

        # 척추체 별로 4-point grouping
        grouped = defaultdict(dict)
        for key, coord in keypoints_dict4.items():
            if coord is not None:
                if self.check_flipped(coord) and coord is not None:
                    print("flip!")
                    w, ori_h = image_result.shape[:2]
                    coord = self.flip(w, ori_h, 'list', coord, 1)
                    image_result = self.flip(w, ori_h, 'image', image_result, 1)

            prefix, index = key.split('-')
            grouped[prefix][index] = coord

        # start inference 6-points
        final_result = []
        for prefix, points in grouped.items():
            if all(points.get(str(i)) is not None for i in range(1, 5)):
                coords = [points[str(i)] for i in range(1, 5)]
                
                final_result.append([coords, prefix])
                
        # final_result.append([grouped])
        final_result = []
        for prefix, points in grouped.items():
            if all(points.get(str(i)) is not None for i in range(1, 5)):
                coords = [points[str(i)] for i in range(1, 5)]
                x_vals = [pt[0] for pt in coords]
                y_vals = [pt[1] for pt in coords]

                x_min, y_min, x_max, y_max = min(x_vals), min(y_vals), max(x_vals), max(y_vals)

                bbox_crop = [x_min, y_min, x_max - x_min, y_max - y_min] # 1개의 척추체 crop

                image_result, preds, labels, scores = self.KEY.keypoint_detect2(image_result, 3, bbox_crop, 288, 384, 0)

                if preds[2][0] > preds[3][0]:
                    preds[2], preds[3] = preds[3], preds[2]
                
                six_points = preds
                # middle_points = preds[-2:]
                # six_points = list(points.values()) + middle_points
                
                final_result.append([six_points, prefix])
            
        return image_result, final_result
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
    # classes = ['Normal', 'Acute', 'Chronic', 'VP']
    # for cls in classes:
        # (out_folder / cls).mkdir(parents=True, exist_ok=True)
    # (out_folder / "Etc").mkdir(parents=True, exist_ok=True)

    # 엑셀 저장 리스트
    summary_results = []

    print(f"Processing {len(image_files)} images...")
    
    for image_path in tqdm(image_files):
        try:
            folder_name = image_path.parent.name
            file_name = image_path.stem
            file_name_ext = image_path.name
            reg_id_raw = folder_name.split('_')[0]
            reg_id = str(int(reg_id_raw))
            
            # 이미지 로드
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                image = prep.convert_dicom2img(image_path)
            else:
                image = cv2.imread(str(image_path))
                if image.ndim == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                    
            image_copy = image.copy()
            h, w = image.shape[:2]
                    
            json_data = MLM.makeStructure(image_copy, file_name_ext)

            json_data['shapes'] = []
            group_id = 1
            
            image_result, point_results = model.inference_6point(image_copy)
            
            for point_info in point_results:
                six_points = point_info[0]
                prefix = point_info[1]
                
                v_name_upper = prefix.upper()
                
                # T10 이상 (위의) 척추체 제외 (T9, T8, T7...)
                should_exclude = False
                
                # 1. 'T'로 시작하는 경우 숫자 확인
                v_num = int(v_name_upper[1:]) 
                
                if v_name_upper.startswith('T'):
                    try:
                        if v_num <= 10:
                            should_exclude = True
                    except ValueError:
                        pass # 숫자가 아닌 형식이면 무시 (혹은 안전하게 제외)
                
                if should_exclude:
                    continue
                
                if len(six_points) < 6: 
                    continue
                
                # 라벨 확인
                if (reg_id, v_name_upper) in gt_map:
                    diagnosis = gt_map[(reg_id, v_name_upper)]
                else:
                    diagnosis = 'Normal'
                    
                x_vals = [p[0] for p in six_points]
                y_vals = [p[1] for p in six_points]
                pad = 10
                x_min = max(0, int(min(x_vals)) - pad)
                y_min = max(0, int(min(y_vals)) - pad)
                x_max = min(w, int(max(x_vals)) + pad)
                y_max = min(h, int(max(y_vals)) + pad)
                
                bbox_label = f"{v_name_upper}_{diagnosis}"
                
                bbox = [x_min, y_min, x_max, y_max]
                
                json_data = MLM.addBox(json_data, bbox, bbox_label, v_num)
                
                for idx, point_coord in enumerate(six_points):
                    json_data = MLM.addKeypoints(json_data, [point_coord], [f'{prefix}-{idx+1}'], v_num)
                    # group_id += 1
                
            shutil.copy2(image_path, out_folder / file_name_ext)
            MLM.saveJson(json_data, out_folder, file_name)
            
        except Exception as e:
            print(f"Error: {image_path.name} - {e}")

        
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