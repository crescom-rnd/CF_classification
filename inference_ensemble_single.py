from utils import makelabelmeFile, preprocessImage, postprocess
from pathlib import Path
from module1 import Module1
from module2 import Module2
import os
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
import re
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.metrics import confusion_matrix, classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_folder = Path()
output_folder = Path("Result_Ensemble_0225")
output_folder.mkdir(parents=True, exist_ok=True)

SAVE_ERROR_IMAGES = True
ERROR_DIR = output_folder / "error_images"

MAX_SAVE_PER_BUCKET = 15

class HistogramFlattening:
    """논문 방식: 전체 히스토그램 평탄화 (Global Equalization)"""
    def __call__(self, img):
        # 1. Grayscale 변환
        img_np = np.array(img.convert('L'))
        
        # 2. Histogram Equalization 적용 (CLAHE 아님)
        img_eq = cv2.equalizeHist(img_np)
        
        # 3. 모델 입력을 위해 RGB로 재변환
        img_eq = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(img_eq)
    
    
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        padding_img = Image.new(image.mode, (max_wh, max_wh), (0, 0, 0))
        padding_img.paste(image, (p_left, p_top))
        return padding_img

class SingleFoldClassifier:
    def __init__(self, stage1_weight_dir, stage2_weight_dir, fold_idx, apply_vcr_filter=True, vcr_threshold=0.07):
        print(f"[INFO] Initializing Single Fold Model (Fold: {fold_idx})...")
        self.fold_idx = fold_idx
        self.apply_vcr_filter = apply_vcr_filter
        self.vcr_threshold = vcr_threshold
        
        self.preprocess = transforms.Compose([
            HistogramFlattening(),
            SquarePad(),
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # --- Load Stage 1 Model (단일 폴드 1개만) ---
        s1_path = os.path.join(stage1_weight_dir, f"best_model_fold_{fold_idx}.pth")
        self.model_s1 = self._load_timm_model(s1_path, num_classes=3)
        
        # --- Load Stage 2 Model (단일 폴드 1개만) ---
        s2_path = os.path.join(stage2_weight_dir, f"best_model_fold_{fold_idx}.pth")
        self.model_s2 = self._load_timm_model(s2_path, num_classes=2)

        self.labels_s2 = {0: 'Acute', 1: 'Chronic'}

    def _load_timm_model(self, path, num_classes):
        model = timm.create_model('convnext_base.fb_in22k_ft_in1k_384', pretrained=False, num_classes=num_classes)
        try:
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"  -> Successfully loaded {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
        model.to(DEVICE)
        model.eval()
        return model

    def predict(self, crop_img_bgr, vcr_ratio):
        if crop_img_bgr is None or crop_img_bgr.size == 0:
            return "Error", "Error", 0.0, "Error", {}

        img_rgb = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.preprocess(img_pil).unsqueeze(0).to(DEVICE)
        
        meta = {}

        # =========================================================
        # [Stage 1] Normal(0) vs Fracture(1) vs VP(2)
        # =========================================================
        with torch.no_grad():
            out_s1 = self.model_s1(input_tensor)
            prob_s1 = torch.softmax(out_s1, dim=1)[0]
        
        p_normal = prob_s1[0].item()
        p_fracture = prob_s1[1].item()
        p_vp = prob_s1[2].item()
        
        meta.update({
            "S1_pNormal": p_normal,
            "S1_pFrac": p_fracture,
            "S1_pVP": p_vp
        })
        
        THRESHOLD_STAGE1 = 0.12

        # 1순위: VP 확실하면 컷
        if p_vp > 0.5:
            return "VP", "VP", p_vp, "Stage1_VP", meta
        
        # 2순위: 골절 확률이 너무 낮으면 Normal 컷
        if p_fracture < THRESHOLD_STAGE1:
            return "Normal", "Normal", p_normal, "Stage1_Normal", meta

        # =========================================================
        # [Stage 2] Acute(0) vs Chronic(1)
        # =========================================================
        with torch.no_grad():
            out_s2 = self.model_s2(input_tensor)
            prob_s2 = torch.softmax(out_s2, dim=1)[0]
        
        p_acute = prob_s2[0].item()
        p_chronic = prob_s2[1].item()
        
        pred_idx = torch.argmax(prob_s2).item()
        dl_pred = self.labels_s2[pred_idx]
        dl_conf = prob_s2[pred_idx].item()
        
        meta.update({
            "S2_pAcute": p_acute,
            "S2_pChronic": p_chronic,
        })

        final_pred = dl_pred
        source = f"Stage2_{dl_pred}"

        # =========================================================
        # [Ablation 포인트] VCR Ratio 기반 안전장치 (Post-processing)
        # =========================================================
        if self.apply_vcr_filter:
            if dl_pred == "Chronic" and vcr_ratio < self.vcr_threshold:
                final_pred = "Normal"
                source = f"Stage2_Chronic_Filter_VCR({self.vcr_threshold})"
                
            # 필요하다면 Acute 필터도 여기서 추가 가능
            # if dl_pred == "Acute" and vcr_ratio < (self.vcr_threshold - 0.02):
            #     final_pred = "Normal"
            #     source = f"Stage2_Acute_Filter_VCR"

        return dl_pred, final_pred, dl_conf, source, meta
    
# MAIN ========================================== 
POS = 'lat'
MLM = makelabelmeFile()
PREP = preprocessImage()
POST = postprocess()
MOD1 = Module1(POS.upper())
MOD2 = Module2()

def load_gt_data(gt_file_path):
    gt_dict = {}
    if not gt_file_path or not os.path.exists(gt_file_path): return gt_dict
    try:
        df = pd.read_csv(gt_file_path) if str(gt_file_path).endswith('.csv') else pd.read_excel(gt_file_path)
        for _, row in df.iterrows():
            gt_dict[(str(row.get('RegID')), str(row.get('Vertebra', '')).lower())] = row.get('Diagnosis', 'Unknown')
    except: pass
    return gt_dict

def inference(image_list, output_folder, classifier, gt_path=None):
    gt_map = load_gt_data(gt_path)
    flat_result = []
    y_true, y_pred = [], []
    TARGET_CLASSES = ['Acute', 'Chronic', 'Normal', 'VP']
    
    print(f"[INFO] Start Ensemble Inference on {len(image_list)} images...")

    for image_file in tqdm(image_list):
        if str(image_file).endswith(".dcm"): image = PREP.convert_dicom2img(image_file)
        else: image = cv2.imread(str(image_file))
        if image is None: continue
        if image.ndim == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        image_copy = image.copy()
        folder = image_file.parent.name
        image_file_name = image_file.name
        m = re.search(r"(\d+)", image_file_name)
        # reg_id = int(m.group(1)) if m else image_file_name 
        reg_id = m.group(1) if m else image_file_name
        reg_id = str(reg_id).lstrip("0") or "0"
        
        file_name_jpg = image_file.stem + ".jpg"
        json_data = MLM.makeStructure(image_copy, file_name_jpg)
        json_data['shapes'] = []

        # Detection Module
        image_result, point_results = MOD1.inference_6point(image_copy)
        image_result_copy = image_result.copy()

        for point_info in point_results:
            six_points, prefix = point_info[0], point_info[1]
            if len(six_points) < 6: continue
            if prefix.upper().startswith('T') and int(prefix[1:]) <= 10: continue

            for idx, p in enumerate(six_points):
                json_data = MLM.addKeypoints(json_data, [p], [f'{prefix}-{idx+1}'], prefix)
        
        # VCR Calculation
        vcr_result, _ = MOD2.compute_vcr(json_data)
        
        for spine_info in vcr_result:
            vb_name = spine_info['vb_name']
            pts = spine_info['points']
            max_vcr = max(spine_info['vcr_a'], spine_info['vcr_m'], spine_info['vcr_p'])
            vcr_ratio = max_vcr / 100.0

            # Crop
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            x_min, y_min = max(0, min(xs)), max(0, min(ys))
            x_max, y_max = min(image.shape[1], max(xs)), min(image.shape[0], max(ys))
            crop_img = image[int(y_min):int(y_max), int(x_min):int(x_max)]
            bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
            
            # Predict (Ensemble)
            dl_pred, final_pred, dl_conf, source, meta = classifier.predict(crop_img, vcr_ratio)

            # Save Result
            gt_key = (str(reg_id), vb_name.lower())
            gt_label = gt_map.get(gt_key, "Unknown")
            
            if gt_label in TARGET_CLASSES:
                y_true.append(gt_label)
                y_pred.append(final_pred)
            
            spine_info.update({
                'RegID': reg_id, 
                'DL_Pred': dl_pred, 
                'vcr_Pred': spine_info.get('vcr_pred'),
                # "S1_pNormal": meta.get('S1_pNormal'),
                # "S1_pFrac": meta.get('S1_pFrac'),
                'Source': source, 
                'Conf': round(dl_conf, 4),
                'vcr_ratio': vcr_ratio,
                "S1_pVP": meta.get('S1_pVP'),
                'S1_Mean_pFrac': meta.get('S1_pFrac'),
                'S2_Mean_pAcute': meta.get('S2_pAcute'),
                'GT_Label': gt_label, 
                'bbox': bbox, 
            })
            
            
            flat_result.append(spine_info)
            # json_data = MLM.addBox(json_data, bbox, label=final_pred, group_id=vb_name)
            # POST.put_text(image_result_copy, spine_info) 

        # MLM.saveJson(json_data, output_folder, image_file.stem)
        # cv2.imwrite(str(output_folder / f'{image_file.stem}.jpg'), cv2.hconcat([image_copy, image_result_copy]))
        # ---- error image save (최종 파이프라인 기준) ----
        # (inference 함수의 for문 내부, Save Result 밑부분)
        
        # ---- Error Image Save ----
        if SAVE_ERROR_IMAGES and gt_label in TARGET_CLASSES and final_pred in TARGET_CLASSES:
            if gt_label != final_pred:
                # 폴드 구분을 위해 속성에 fold_idx 추가된 거 활용
                fold_num = getattr(classifier, 'fold_idx', 'Ensemble')
                bucket = f"{gt_label}_to_{final_pred}"
                save_dir = ERROR_DIR / f"Fold_{fold_num}" / bucket
                save_dir.mkdir(parents=True, exist_ok=True)

                if len(list(save_dir.glob("*.jpg"))) < MAX_SAVE_PER_BUCKET:
                    # 파일명에 dl_pred(딥러닝 원본 예측)와 final_pred(VCR 필터 거친 예측) 둘 다 표기
                    save_path = save_dir / f"{reg_id}_{vb_name}_GT-{gt_label}_DL-{dl_pred}_PR-{final_pred}_vcr{vcr_ratio:.3f}.jpg"
                    vis = image_result_copy.copy()

                    x1,y1,x2,y2 = bbox
                    cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2) # 틀린건 빨간 박스

                    txt1 = f"GT: {gt_label} | DL: {dl_pred} -> Final: {final_pred}"
                    txt2 = f"VCR: {vcr_ratio:.3f} | Conf: {dl_conf:.2f} | Src: {source}"
                    
                    cv2.putText(vis, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)
                    cv2.putText(vis, txt2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(vis, txt2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1, cv2.LINE_AA)

                    cv2.imwrite(str(save_path), vis)

    pd.DataFrame(flat_result).to_excel(output_folder / 'final_result_ensemble.xlsx', index=False)
    
    if len(y_true) > 0:
        # print("\n[Ensemble Confusion Matrix]")
        # print(confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES))
        # print(classification_report(y_true, y_pred, labels=TARGET_CLASSES, digits=4))
        if len(y_true) > 0:
            cm = confusion_matrix(y_true, y_pred, labels=TARGET_CLASSES)
            print("\n[Ensemble Confusion Matrix]")
            print(cm)
            
            # --- 의료용 지표 계산 (Medical Metrics) ---
            # TARGET_CLASSES = ['Acute', 'Chronic', 'Normal', 'VP']
            # 인덱스: Acute(0), Chronic(1), Normal(2), VP(3)
            
            # 1. Acute Sensitivity (급성 골절 민감도)
            # Acute를 Acute라고 맞춘 것 / 실제 Acute 전체
            acute_tp = cm[0,0]
            acute_fn = cm[0,1] + cm[0,2] + cm[0,3] # Chronic, Normal, VP로 잘못 예측한 것
            acute_sensitivity = acute_tp / (acute_tp + acute_fn + 1e-6)
            
            # 2. Chronic Sensitivity (만성 골절 민감도)
            chronic_tp = cm[1,1]
            chronic_fn = cm[1,0] + cm[1,2] + cm[1,3]
            chronic_sensitivity = chronic_tp / (chronic_tp + chronic_fn + 1e-6)
            
            # 3. Normal Specificity (정상 특이도) - 가장 중요!
            # Normal을 Normal이라고 맞춘 것 / 실제 Normal 전체
            normal_tn = cm[2,2]
            normal_fp = cm[2,0] + cm[2,1] + cm[2,3] # Normal을 골절(Acute/Chronic/VP)로 오진한 것
            normal_specificity = normal_tn / (normal_tn + normal_fp + 1e-6)
            
            # 4. VP Sensitivity
            vp_tp = cm[3,3]
            vp_fn = cm[3,0] + cm[3,1] + cm[3,2]
            vp_sensitivity = vp_tp / (vp_tp + vp_fn + 1e-6)

            print("\n" + "="*60)
            print("          🏥 MEDICAL METRICS REPORT (Sensitivity & Specificity) 🏥")
            print("="*60)
            
            # TARGET_CLASSES = ['Acute', 'Chronic', 'Normal', 'VP']
            # 인덱스: 0:Acute, 1:Chronic, 2:Normal, 3:VP
            
            # ---------------------------------------------------------
            # 1. 각 클래스별 One-vs-Rest 민감도/특이도 계산 함수
            # ---------------------------------------------------------
            def get_binary_metrics(cm, idx):
                # TP: 해당 클래스를 정확히 맞춤
                tp = cm[idx, idx]
                
                # FN: 해당 클래스인데 다른 걸로 예측함 (Row의 합 - TP)
                fn = np.sum(cm[idx, :]) - tp
                
                # FP: 다른 클래스인데 해당 클래스로 예측함 (Col의 합 - TP)
                fp = np.sum(cm[:, idx]) - tp
                
                # TN: 해당 클래스도 아니고, 예측도 다른 걸로 함 (나머지 전체)
                tn = np.sum(cm) - (tp + fn + fp)
                
                sensitivity = tp / (tp + fn + 1e-6)
                specificity = tn / (tn + fp + 1e-6)
                
                return sensitivity, specificity, tp, fn, tn, fp

            # ---------------------------------------------------------
            # 2. 클래스별 상세 리포트 출력
            # ---------------------------------------------------------
            print(f"{'Class':<10} | {'Sensitivity (민감도)':<20} | {'Specificity (특이도)':<20}")
            print("-" * 60)
            
            # 각 클래스 순회
            metrics_dict = {}
            for idx, class_name in enumerate(TARGET_CLASSES):
                sens, spec, tp, fn, tn, fp = get_binary_metrics(cm, idx)
                metrics_dict[class_name] = {'sens': sens, 'spec': spec}
                print(f"{class_name:<10} | {sens:.2%} ({tp}/{tp+fn})   | {spec:.2%} ({tn}/{tn+fp})")

            print("-" * 60)

            # ---------------------------------------------------------
            # 3. [대표님 보고용] 골절(Fracture) 통합 지표
            # ---------------------------------------------------------
            # Normal(인덱스 2)을 제외한 모든 것을 "골절"로 간주
            # 골절 민감도 = (실제 골절인데 골절(Acute/Chronic/VP)로 예측한 수) / 실제 골절 총합
            # 골절 특이도 = (실제 정상인데 정상으로 예측한 수) / 실제 정상 총합 (= Normal Class의 민감도와 동일)
            
            # 실제 Normal 개수 (Support)
            total_normal = np.sum(cm[2, :])
            # Normal을 Normal로 맞춘 개수 (TN for Fracture)
            correct_normal = cm[2, 2]
            
            # 실제 Fracture 개수 (Acute + Chronic + VP)
            # 0, 1, 3번 행의 합
            total_fracture = np.sum(cm) - total_normal
            
            # Fracture를 Fracture(Acute/Chronic/VP 중 하나)로 맞춘 개수
            # 전체 TP - Normal을 Normal로 맞춘 것 + Normal을 오진한 것... 
            # 더 쉽게: (전체 예측 - Normal로 예측된 것) 중 실제 Fracture 인 것? 복잡하니 단순 합산
            
            # 실제 Fracture(행 0,1,3) 중 예측도 Fracture(열 0,1,3) 인 것의 합
            correct_fracture_detection = 0
            fracture_indices = [0, 1, 3] # Acute, Chronic, VP
            for r in fracture_indices:
                for c in fracture_indices:
                    correct_fracture_detection += cm[r, c]

            sys_sensitivity = correct_fracture_detection / (total_fracture + 1e-6)
            sys_specificity = correct_normal / (total_normal + 1e-6) # = Normal Recall

            print("\n[📢 EXECUTIVE SUMMARY (골절 vs 정상)]")
            print(f"1. Total Fracture Sensitivity (전체 골절 민감도) : {sys_sensitivity:.2%}  <-- 환자를 놓치지 않는 확률")
            print(f"2. Total Normal Specificity   (전체 정상 특이도) : {sys_specificity:.2%}  <-- 정상을 정상이라 할 확률")
            print("="*60)

def parse_regid_from_filename(p: Path) -> str:
    # 파일명 예: 0001_lat00001.jpg  -> "1" 또는 "0001"로 맞출 수 있음
    name = p.stem  # 0001_lat00001
    m = re.match(r"(\d+)", name)
    if not m:
        return ""
    raw = m.group(1)          # "0001"
    # CSV RegID가 1,2,3.. 형태면 앞의 0 제거해서 맞추기
    regid = raw.lstrip("0") or "0"
    return regid

def build_image_list_for_fold(input_folder: Path, regid_set: set[str]):
    imgs = []
    for p in input_folder.rglob("*"):
        if p.suffix.lower() not in [".jpg", ".png", ".jpeg", ".dcm", ".dicom"]:
            continue
        rid = parse_regid_from_filename(p)
        if rid and rid in regid_set:
            imgs.append(p)
    return sorted(imgs)

def get_val_regids_for_fold(csv_path: str, fold: int) -> set[str]:
    df = pd.read_csv(csv_path)
    # CSV도 str로 통일
    df["RegID"] = df["RegID"].astype(str).str.strip()
    # fold==k 인 애들만 (너 fold 정의가 "그 fold가 val"로 쓰는 거 맞지?)
    regids = set(df.loc[df["fold"] == fold, "RegID"].unique().tolist())
    return regids

if __name__ == "__main__":
    # 경로 설정
    MODE = 'train'
    csv_path = "crop_summary_with_folds.csv" # k-fold 해 놓은 csv

    # gt_path = Path("cropping_vertebra_result_from_json_lat_test/test_list.csv")
    gt_path = csv_path
    input_folder = root_folder / f"Converting_from_dcm_{MODE}"
    
    # 예: result_weights/kfold_stage1_maxvit/ 폴더 안에 best_model_fold_0.pth ... 가 있어야 함
    stage1_dir = "result_weights/patient_kfold/STAGE1_convnext_base.fb_in22k_ft_in1k_384" 
    stage2_dir = "result_weights/patient_kfold/STAGE2_convnext_base.fb_in22k_ft_in1k_384"
    
    n_folds = 5
    
    # 💡 Ablation Study 컨트롤러
    APPLY_VCR_FILTER = True
    VCR_THRESHOLD = 0.07
    
    # OOF 결과를 한 번에 모아서 보기 위한 리스트
    oof_y_true = []
    oof_y_pred = []

    for fold in range(n_folds):
        output_folder = Path(f"Result_OOF_VAL_FOLD{fold}")
        output_folder.mkdir(parents=True, exist_ok=True)

        regid_set = get_val_regids_for_fold(csv_path, fold)
        print(f"\n======================================")
        print(f"[Fold {fold}] val patients = {len(regid_set)}")
        print(f"======================================")

        image_list = build_image_list_for_fold(input_folder, regid_set)
        
        # 핵심: SingleFoldClassifier 호출, 파라미터 전달
        classifier = SingleFoldClassifier(
            stage1_dir, stage2_dir, fold_idx=fold, 
            apply_vcr_filter=APPLY_VCR_FILTER, vcr_threshold=VCR_THRESHOLD
        )

        # inference 함수 실행 후 리턴값을 받아서 통합 OOF 성능을 잴 수도 있습니다.
        # (현재 inference 내부에서 자체 프린트하게 두셔도 됩니다)
        inference(image_list, output_folder, classifier, gt_path=Path(csv_path))

    # image_list = []
    # for p in input_folder.rglob("*"):
    #     if p.suffix.lower() in [".dcm", ".dicom", ".jpg", ".png"]:
    #         image_list.append(p)
            
    # # 5개의 Fold가 다 학습된 후에 돌리세요!
    # # 아직 학습 중이라면, 학습 끝날 때까지 기다려야 합니다.
    # classifier = EnsembleFractureClassifier(stage1_dir, stage2_dir, n_folds=5)
    
    # inference(sorted(image_list), output_folder, classifier, gt_path=gt_path)