import math
from pathlib import Path
from scipy.spatial import distance
from scipy.special import expit
from collections import defaultdict
import cv2
from matplotlib import pyplot as plt
import copy
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F


class Module2:
    def __init__(self):
        self.vertebrae = ['t1', 't2', 't3', 't4', 't5',
                    't6', 't7', 't8', 't9', 't10',
                    't11', 't12', 'l1', 'l2', 'l3',
                    'l4', 'l5']
        self.point_num = 6
        self.label_keys = [f"{ver}-{pn+1}" for ver in self.vertebrae for pn in range(self.point_num)]
        self.ratio_threshold = 20
        self.vcr_threshold = 1
        self.shape_threshold = 0
        self.total_threshold = 0.5
        # 0717 추가
        self.xvfa_tolerance = {"apr": [0.2, 0.25, 0.4],
                          "mpr": [0.2, 0.25, 0.4],
                          "mar": [0.2, 0.25, 0.4]}
        self.xvfa_threshold = {"apr": 1.0, "mpr": 1.0, "mar": 1.0}


    def get_group_label(self, key_dict):
        temp = defaultdict(dict)
        for k, v in key_dict.items():
            prefix, index = k.split("-")
            temp[prefix][index] = v

        # 점 6개 존재하고 None이 아닌 경우를 필터링
        result = {
            prefix: {str(i): pts[str(i)] for i in range(1, 7)}
            for prefix, pts in temp.items()
            if all(pts.get(str(i)) is not None for i in range(1, 7))
        }

        # result: 
        return result
    

    def match_label_coord(self, json_data, labels):
        key_dict = {key : None for key in labels}
        for shape in json_data['shapes']:
            if shape['label'] in key_dict.keys():
                if shape['shape_type'] == 'point':
                    key_dict[shape['label']] = shape['points'][0]

        return key_dict
    

    def compute_soft_vcr_module(self, vb_name, points, ha, hm, hp, ref_a, ref_m, ref_p, scale=0.3):
        vcr_a = round((1 - (ha / ref_a)) * 100, 2) if ref_a > 0 else 0
        vcr_m = round((1 - (hm / ref_m)) * 100, 2) if ref_m > 0 else 0
        vcr_p = round((1 - (hp / ref_p)) * 100, 2) if ref_p > 0 else 0

        # soft sigmoid score
        x = torch.tensor([vcr_a, vcr_m, vcr_p])
        score = torch.sigmoid(scale * (x - self.ratio_threshold))
        vcr_prob = round(torch.max(score).item(), 4)

        vcr_cf = "Fracture" if vcr_prob > self.total_threshold else "Normal"

        return vcr_a, vcr_m, vcr_p, vcr_cf, vcr_prob

    def compute_vcr_module(self, cu_ha, cu_hm, cu_hp, ref_a, ref_m, ref_p):
        vcr_a = round((1 - (cu_ha / ref_a)) * 100, 2) if ref_a > 0 else 0
        vcr_m = round((1 - (cu_hm / ref_m)) * 100, 2) if ref_m > 0 else 0
        vcr_p = round((1 - (cu_hp / ref_p)) * 100, 2) if ref_p > 0 else 0

        max_vcr = max(vcr_a, vcr_m, vcr_p)

        # TODO: max값과 나머지 값이 n%내에서 비슷한 경우도 고려해야 함
        # -> tolerance 허용오차

        threshold = 20 # 20% 이상이면 압박골절
        tolerance = 0 # 0%
        lower_bound = threshold - tolerance

        if max_vcr > lower_bound:
            if max_vcr == vcr_a:
                vcr_cf = 'Wedge'
            elif max_vcr == vcr_m:
                vcr_cf = 'Concave'
            elif max_vcr == vcr_p:
                vcr_cf = 'Crush'
        elif 17 < max_vcr < threshold:
            vcr_cf = 'Suspected'
        else:
            vcr_cf = 'Normal'

        return vcr_a, vcr_m, vcr_p, vcr_cf    

    def compute_shape_ratio(self, ha, hm, hp, tol=0.2):
        # ha, hm, hp = cu_vb['ha'], cu_vb['hm'], cu_vb['hp'],
        ha = torch.tensor([ha], dtype=torch.float32)
        hm = torch.tensor([hm], dtype=torch.float32)
        hp = torch.tensor([hp], dtype=torch.float32)
        mask_hp = hp > 0
        APR = torch.where(mask_hp, ha / hp, torch.tensor(0.0, dtype=torch.float32))

        mask_ha = ha > 0
        MAR = torch.where(mask_ha, hm / ha, torch.tensor(0.0, dtype=torch.float32))

        mask_hp2 = hp > 0
        MPR = torch.where(mask_hp2, hm / hp, torch.tensor(0.0, dtype=torch.float32))

        # APR, MPR, MAR = round(APR, 4), round(MPR, 4), round(MAR, 4)

        # Genant’s rule-based fracture type prediction
        # XVFA 참조
        apr_pos = self.geq(APR, self.xvfa_threshold["apr"])
        mpr_pos = self.geq(MPR, self.xvfa_threshold["mpr"])
        mar_pos = self.geq(MAR, self.xvfa_threshold["mar"])

        apr_neg = self.leq(APR, self.xvfa_threshold["apr"])
        mpr_neg = self.leq(MPR, self.xvfa_threshold["mpr"])
        mar_neg = self.leq(MAR, self.xvfa_threshold["mar"])

        normal = self.within(APR, MPR, MAR, tolerance_idx=0)

        crush, ind = torch.stack([
            mpr_pos, 
            mar_neg, 
            apr_pos, 
            1-normal, 
            ], dim=1).min(dim=1) # and

        biconcave, ind = torch.stack([
            mpr_neg, 
            mar_neg, 
            1-normal, 
            1-crush, 
            ], dim=1).min(dim=1) # and

        wedge, ind = torch.stack([
            mpr_neg, 
            mar_pos, 
            apr_neg, 
            1-normal, 
            1-crush, 
            1-biconcave
            ], dim=1).min(dim=1) # and
        
        type_logits = torch.stack([normal, wedge, biconcave, crush], dim=-1) 
        return APR, MPR, MAR, type_logits
    
        # 1-ratio가 0에 가까우면 정상 (ratio가 1에 가까우면 정상)
        # if all(1 - tol <= r <= 1 + tol for r in [APR, MPR, MAR]):
        #     fracture_shape = "Normal"
        # elif MPR < 1:
        #     if MAR > 1:
        #         # fracture_shape = "Wedge"
        #         fracture_shape = "Fracture"
        #     elif MAR < 1:
        #         # fracture_shape = "Concave"
        #         fracture_shape = "Fracture"
        #     else:
        #         # fracture_shape = "Crush"
        #         fracture_shape = "Fracture"
        # elif MPR >= 1 and MAR <= 1:
        #     # fracture_shape = "Convex"
        #     fracture_shape = "Fracture"
        # else:
        #     fracture_shape = "Uncertain"

        # # 확률 기반 score (선형 스코어 방식 사용)
        # deviation = sum([abs(r - 1) for r in [APR, MPR, MAR]])
        # max_deviation = 3 * tol  # 최대 허용 오차
        # shape_prob = max(0, 1 - deviation / max_deviation)
        # shape_prob = round(shape_prob, 4)

        # return APR, MPR, MAR, fracture_shape, shape_prob
    
    # 0717 추가
    def geq(self, x, value):
        """
        x (tensor): value to compare
        value (tensor) : value to compare against
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)

        # Broadcasting-safe
        if value.dim() == 0:
            value = value.expand_as(x)

        return F.sigmoid(x - value)
    
    def leq(self, x, value):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)

        # Broadcasting-safe
        if value.dim() == 0:
            value = value.expand_as(x)
        return F.sigmoid(value - x)
    
    def within(self, apr, mpr, mar, tolerance_idx: int = 1):
        """
        Fuzzy approximation to check if the vertebrae is within the given tolerance.
        
        Args:
            apr (Tensor): Anterior/posterior ratio (N,1)
            mpr (Tensor): Middle/posterior ratio (N,1)
            mar (Tensor): Middle/anterior ratio (N,1)
            tolerance_idx (int): Index of the tolerance to use.
        """

        apr_pos_thresh = self.xvfa_threshold["apr"]*(1-self.xvfa_tolerance["apr"][tolerance_idx])
        mpr_pos_thresh = self.xvfa_threshold["mpr"]*(1-self.xvfa_tolerance["mpr"][tolerance_idx])
        mar_pos_thresh = self.xvfa_threshold["mar"]*(1-self.xvfa_tolerance["mar"][tolerance_idx])

        apr_neg_thresh = self.xvfa_threshold["apr"]*(1+self.xvfa_tolerance["apr"][tolerance_idx])
        mpr_neg_thresh = self.xvfa_threshold["mpr"]*(1+self.xvfa_tolerance["mpr"][tolerance_idx])
        mar_neg_thresh = self.xvfa_threshold["mar"]*(1+self.xvfa_tolerance["mar"][tolerance_idx])

        is_within, _ = torch.stack([
            self.geq(apr, apr_pos_thresh), 
            self.geq(mpr, mpr_pos_thresh), 
            self.geq(mar, mar_pos_thresh), 
            self.leq(apr, apr_neg_thresh), 
            self.leq(mpr, mpr_neg_thresh), 
            self.leq(mar, mar_neg_thresh)
            ], dim=1).min(dim=1)
        
        return is_within
    #### 

    def axis_heights(self, group_vb):
        spine_info = []
        vb_list = list(group_vb)

        for i in range(len(vb_list)):
            vb_name = vb_list[i]
            points = [[int(x) for x in v] for v in group_vb[vb_name].values()]

            h_anterior = distance.euclidean(points[0], points[2])
            h_middle = distance.euclidean(points[4], points[5])
            h_posterior = distance.euclidean(points[1], points[3])

            infos = {
                    'vb_name': vb_name,
                    'ha': h_anterior,
                    'hm': h_middle,
                    'hp': h_posterior,
                    'points': points
                }
            
            spine_info.append(infos)

        return spine_info
    

    def gathering_result(self, vb_name, points, ha, hm, hp, ref_a, ref_m, ref_p):
        result = []

        vcr_a, vcr_m, vcr_p, vcr_cf = self.compute_vcr_module(ha, hm, hp, ref_a, ref_m, ref_p)
        # vcr_a, vcr_m, vcr_p, vcr_cf, vcr_prob = self.compute_soft_vcr_module(vb_name, points, ha, hm, hp, ref_a, ref_m, ref_p)
        # APR, MPR, MAR, fracture_shape, shape_prob = self.compute_shape_ratio(ha, hm, hp)

        APR, MPR, MAR, type_logits = self.compute_shape_ratio(ha, hm, hp)
        shape_prob = float(type_logits.max())
        type_labels = ["Normal", "Wedge", "Concave", "Crush"]
        pred_idx = type_logits.argmax(dim=1)
        shape_cf = [type_labels[i] for i in pred_idx][0]
        
        # total_prob = round(self.vcr_threshold * vcr_prob + self.shape_threshold * shape_prob, 4)
        # final_pred = 'Fracture' if total_prob > self.total_threshold else 'Normal'
        vcr_pred = 'Fracture' if vcr_cf in {"Wedge", "Crush", "Concave", "Fracture"} else vcr_cf

        result = {
            'vb_name': vb_name,
            'points': points,
            'ha': ha,
            'hm': hm,
            'hp': hp,
            'vcr_a': vcr_a,
            'vcr_m': vcr_m,
            'vcr_p': vcr_p,
            'vcr_cf': vcr_cf,
            # 'vcr_prob': vcr_prob,
            'APR': APR,
            'MPR': MPR,
            'MAR': MAR,
            'shape_cf': shape_cf,
            'shape_prob': shape_prob,
            # 'total_prob': total_prob,
            'vcr_pred': vcr_pred,
        }

        return result


    def compute_init(self, spine_info):
        vcr_result = []

        for i, cu_vb in enumerate(spine_info):
            up_vb = spine_info[i - 1] if i > 0 else None
            down_vb = spine_info[i + 1] if i < len(spine_info) - 1 else None

            if up_vb and down_vb:
                ref_a = (up_vb['ha'] + down_vb['ha']) / 2
                ref_m = (up_vb['hm'] + down_vb['hm']) / 2
                ref_p = (up_vb['hp'] + down_vb['hp']) / 2
            elif up_vb:
                ref_a, ref_m, ref_p = up_vb['ha'], up_vb['hm'], up_vb['hp']
            elif down_vb:
                ref_a, ref_m, ref_p = down_vb['ha'], down_vb['hm'], down_vb['hp']
            else:
                continue

            get_result = self.gathering_result(cu_vb['vb_name'], cu_vb['points'], cu_vb['ha'], cu_vb['hm'], cu_vb['hp'], ref_a, ref_m, ref_p)
            # get_result = self.compute_soft_vcr_module(cu_vb['vb_name'], cu_vb['points'], cu_vb['ha'], cu_vb['hm'], cu_vb['hp'], ref_a, ref_m, ref_p)
            vcr_result.append(get_result)

        return vcr_result
    

    def re_compute(self, vcr_result):
        result = []
        for i, cu_vb in enumerate(vcr_result):
            # if cu_vb['vb_name'] == 't11':
            #     print()

            candidates = []

            # 위쪽 방향 탐색
            for offset in range(1, 3):
                idx = i - offset
                if idx >= 0:
                    if vcr_result[idx]['vcr_pred'] == 'Normal':
                    # if vcr_result[idx]['vcr_pred'] in ['Normal', 'Suspected']:
                        candidates.append(vcr_result[idx])
                        break # 위쪽에서 한개만 고르기

            # 아래쪽 방향 탐색
            for offset in range(1, 3):
                idx = i + offset
                if idx < len(vcr_result):
                    if vcr_result[idx]['vcr_pred'] == 'Normal':
                    # if vcr_result[idx]['vcr_pred'] in ['Normal', 'Suspected']:
                        candidates.append(vcr_result[idx])
                        break

            if not candidates:
                result.append(cu_vb)
                continue

            # 정상 척추체 기준 높이
            ref_a = sum(vb['ha'] for vb in candidates) / len(candidates)
            ref_m = sum(vb['hm'] for vb in candidates) / len(candidates)
            ref_p = sum(vb['hp'] for vb in candidates) / len(candidates)
            
            get_result = self.gathering_result(cu_vb['vb_name'], cu_vb['points'], cu_vb['ha'], cu_vb['hm'], cu_vb['hp'], ref_a, ref_m, ref_p)
            # get_result = self.gathering_result(cu_vb['vb_name'], cu_vb['points'], cu_vb['ha'], cu_vb['hm'], cu_vb['hp'], ref_a, ref_m, ref_p)
            # get_result = self.compute_soft_vcr_module(cu_vb['vb_name'], cu_vb['points'], cu_vb['ha'], cu_vb['hm'], cu_vb['hp'], ref_a, ref_m, ref_p)
            result.append(get_result)

        return result

    def compute_vcr(self, json_data):
        """
        return: list of vertebrae
                'vb_name': each vertebral name,
                'pred_cf': compress fracture type,
                'points': 6 points,
                'ha': height of anterior,
                'hm': height of middle,
                'hp': height of posterior,
                'vcr_a': vcr of anterior,
                'vcr_m': vcr of middle,
                'vcr_p': vcr of posterior
        """
        key_dict = self.match_label_coord(json_data, self.label_keys)

        # 척추체 별 grouping
        group_vb = self.get_group_label(key_dict)

        # 1. 척추체 높이 구하기
        spine_info = self.axis_heights(group_vb)

        # 2. 인접 척추체들의 높이와 압박률 계산하기
        final_result = self.compute_init(spine_info)
        
        # 중간 결과 저장 리스트 1
        inter_result = []
        inter_result.append([
            dict(item, step=0) for item in copy.deepcopy(final_result)
        ])

        # 3. 골절 의심 척추체가 있는지 확인 & 4. 주변 재계산
        repeat_count = 3
        for step in range(repeat_count):
            final_result = self.re_compute(final_result)
            inter_result.append([
                dict(**item, step=step) for item in copy.deepcopy(final_result)
            ])

        # 중간 결과 저장 리스트(dict) 2
        from collections import defaultdict
        merged_result_dict = defaultdict(dict)

        step_keys = ['vcr_a', 'vcr_m', 'vcr_p', 'vcr_cf', 'vcr_pred']
        # step_keys = ['vcr_a', 'vcr_m', 'vcr_p', 'vcr_prob', 'vcr_cf', 'vcr_pred']

        for step, step_result in enumerate(inter_result):
            for item in step_result:
                vb = item['vb_name']
                for k in step_keys:
                    merged_result_dict[vb][f"{k}_step{step}"] = item[k]

                # TODO: 17~20% 압박률은 골절 의심으로 표기 필요
                if step == repeat_count:
                    merged_result_dict[vb].update({
                        'vb_name': vb,
                        'points': item['points'],
                        'ha': item['ha'],
                        'hm': item['hm'],
                        'hp': item['hp'],
                        'APR': item['APR'],
                        'MPR': item['MPR'],
                        'MAR': item['MAR'],
                        'shape_prob': item['shape_prob'],
                        'shape_cf': item['shape_cf'],
                        # 'vcr_prob': item['vcr_prob'],
                        'vcr_cf': item['vcr_cf'],
                        # 'total_prob': item['total_prob'],
                        'vcr_pred': item['vcr_pred'],
                    })
        

        return final_result, merged_result_dict