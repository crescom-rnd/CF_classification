from utils import keypointDetection, makelabelmeFile
from scipy.spatial import distance
from collections import defaultdict
import pydicom
from pydicom.pixel_data_handlers import apply_modality_lut, apply_voi_lut
import numpy as np
import cv2

"""
keymodels = [
    device,
    model,
    key_ckpt,
    key_n,
    img_height,
    img_width,
    labels,
    yaml_file
]
"""

device_num = 1

class Module1:
    def __init__(self, position):
        self.names = [
            's1-1', 's1-2',
            'l5-3', 'l5-4', 'l5-1', 'l5-2',
            'l4-3', 'l4-4', 'l4-1', 'l4-2',
            'l3-3', 'l3-4', 'l3-1', 'l3-2',
            'l2-3', 'l2-4', 'l2-1', 'l2-2',
            'l1-3', 'l1-4', 'l1-1', 'l1-2',
            't12-3', 't12-4', 't12-1', 't12-2',
            't11-3', 't11-4', 't11-1', 't11-2',
            't10-3', 't10-4', 't10-1', 't10-2',
            't9-3', 't9-4', 't9-1', 't9-2',
            't8-3', 't8-4', 't8-1', 't8-2',
            't7-3', 't7-4', 't7-1', 't7-2',
            't6-3', 't6-4', 't6-1', 't6-2',
            't5-3', 't5-4', 't5-1', 't5-2'
        ]
        self.position = position
        self.weight_11 = f'{position}/1st_1'
        self.weight_12 = f'{position}/1st_2'
        self.weight_2 = f'{position}/2nd'
        self.weight_3 = f'{position}/3rd_6'
        self.max_y_delta = 1 if position == 'LAT' else 1/3
        
        self.keymodels = [
            [
                device_num,
                'pose_hrnet',
                f'weight/{self.weight_11}.pth',
                12,
                384,
                288,
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                [f'config/{self.weight_11}.yaml', 'MODEL', 'EXTRA'],      
            ],
            [
                device_num,
                'pose_hrnet',
                f'weight/{self.weight_12}.pth',
                12,
                384,
                288,
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                [f'config/{self.weight_12}.yaml', 'MODEL', 'EXTRA'],     
            ],
            [
                device_num,
                'pose_hrnet',
                f'weight/{self.weight_2}.pth',
                12,
                384,
                288,
                ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'],
                [f'config/{self.weight_2}.yaml', 'MODEL', 'EXTRA'],     
            ],
            # [
            #     device_num,
            #     'pose_hrnet',
            #     'D:/cf_train/train_output_0714/cfl_avg_end/pose_hrnet/w48_384x288_adam_lr1e-3/best_distance_state.pth',
            #     6,
            #     384,
            #     288,
            #     ['0', '1', '2', '3', '4', '5'],
            #     ['D:/cf_train/experiments/compress_fracture/w48_384x288_adam_lr1e-3.yaml', 'MODEL', 'EXTRA']
            # ]
            
        ]
        if position == 'LAT':
            self.keymodels.append(
                [
                device_num,
                'pose_hrnet',
                f'weight/{self.weight_3}.pth',
                6,
                384,
                288,
                ['0', '1', '2', '3', '4', '5'],
                [f'config/{self.weight_3}.yaml', 'MODEL', 'EXTRA'],     
            ]
            )
        self.KEY = keypointDetection()
        self.KEY.keypoint_load_model(self.keymodels)
        self.MLM = makelabelmeFile()


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
        image, coords, labels, scores = self.KEY.keypoint_detect2(image, 0, [0, 0, w, ori_h], 288, 384, 0.1)

        x_distances = []
        for coord_set in zip(coords[::2], coords[1::2]):
            if coord_set[0][0] is not None and coord_set[0][1] is not None and coord_set[1][0] is not None and coord_set[1][1] is not None:
                x_distances.append(distance.euclidean(coord_set[0], coord_set[1]))

        # # TEST 1: flip 여부 확인 후 flip
        # if self.check_flipped(coords):
        #     print("flip!")
        #     coords = self.flip(w, ori_h, 'list', coords, 1)
        #     image = self.flip(w, ori_h, 'image', image, 1)
        
        # 첫번째 추론이 안 된 경우 = coords에 None이 포함되어 있을 경우 대비 처리
        valid_coords = [c for c in coords if c is not None and c[0] is not None and c[1] is not None]
        if not valid_coords:
            pass

        distance_mean = np.mean(x_distances)
        _, y_mean = np.mean(valid_coords, axis=0)
        # _, y_mean = np.mean(coords, axis=0)
        h = ori_h
        
        top_dist = 8 if self.position == 'LAT' else 6
        bot_dist = 6 if self.position == 'LAT' else 5

        if y_mean > distance_mean * top_dist or (ori_h - y_mean) > distance_mean * bot_dist:
            y_min = max(0, y_mean - distance_mean * top_dist)
            y_max = min(ori_h, y_mean + distance_mean * bot_dist)
            h = y_max - y_min
            image, coords, labels, scores = self.KEY.keypoint_detect2(image, 0, [0, y_min, w, h], 288, 384, 0.1)
        else:
            h = ori_h

        for coord in coords:
            if coord[1] is None or coord[1] > ori_h:
                image, coords, labels, scores = self.KEY.keypoint_detect2(image, 1, [0, 0, w, ori_h], 288, 384, 0.1)
                # print('process')
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

        for _ in range(4):
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