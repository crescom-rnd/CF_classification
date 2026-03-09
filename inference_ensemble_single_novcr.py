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
output_folder = Path("Result_Ensemble_P7")
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
    def __init__(self, stage1_weight_dir, stage2_weight_dir, fold_idx):
        print(f"[INFO] Initializing Single Fold Model (Fold: {fold_idx})...")
        self.fold_idx = fold_idx
        
        self.preprocess = transforms.Compose([
            HistogramFlattening(),
            SquarePad(),
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.preprocess_s1 = transforms.Compose([
            HistogramFlattening(),
            SquarePad(),
            transforms.Resize((384, 384)), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.preprocess_s2 = transforms.Compose([
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
        model_name = Path(path).parent.name.split("_", 1)[1]
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
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
            return "Error", "Error", 0.0, "Error", {}, [0.0, 0.0, 0.0, 0.0]

        img_rgb = cv2.cvtColor(crop_img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        input_tensor = self.preprocess(img_pil).unsqueeze(0).to(DEVICE)
        input_tensor_s1 = self.preprocess_s1(img_pil).unsqueeze(0).to(DEVICE)
        input_tensor_s2 = self.preprocess_s2(img_pil).unsqueeze(0).to(DEVICE)
        
        meta = {}

        # =========================================================
        # [Stage 1] Normal(0) vs Fracture(1) vs VP(2)
        # =========================================================
        with torch.no_grad():
            out_s1 = self.model_s1(input_tensor_s1)
            prob_s1 = torch.softmax(out_s1, dim=1)[0]
        
        p_normal = prob_s1[0].item()
        p_fracture = prob_s1[1].item()
        p_vp = prob_s1[2].item()
        
        meta.update({
            "S1_pNormal": p_normal,
            "S1_pFrac": p_fracture,
            "S1_pVP": p_vp,
            "S1_Mean_pFrac": p_fracture # 호환성
        })
        
        THRESHOLD_STAGE1 = 0.12
        # THRESHOLD_STAGE1 = 0.20

        probs = [0.0, 0.0, 0.0, 0.0]

        # 1순위: VP 확실하면 컷
        if p_vp > 0.5:
            probs = [0.0, 0.0, 0.0, p_vp]
            return "VP", "VP", p_vp, "Stage1_VP", meta, probs
        
        # 2순위: 골절 확률이 너무 낮으면 Normal 컷
        if p_fracture < THRESHOLD_STAGE1:
            probs = [0.0, 0.0, p_normal, 0.0]
            return "Normal", "Normal", p_normal, "Stage1_Normal", meta, probs

        # =========================================================
        # [Stage 2] Acute(0) vs Chronic(1)
        # =========================================================
        with torch.no_grad():
            out_s2 = self.model_s2(input_tensor_s2)
            prob_s2 = torch.softmax(out_s2, dim=1)[0]
        
        p_acute = prob_s2[0].item()
        p_chronic = prob_s2[1].item()
        
        THRESHOLD_CHRONIC = 0.42
        
        if p_chronic > THRESHOLD_CHRONIC:
            dl_label = "Chronic"
            dl_conf = p_chronic
        else:
            dl_label = "Acute"
            dl_conf = p_acute

        # pred_idx = torch.argmax(prob_s2).item()
        # dl_label = self.labels_s2[pred_idx]
        # dl_conf = prob_s2[pred_idx].item()
        
        meta.update({
            "S2_pAcute": p_acute,
            "S2_pChronic": p_chronic,
        })

        dl_pred = dl_label
        final_pred = dl_label
        source = f"Stage2_{dl_label}"

        probs = [p_fracture * p_acute, p_fracture * p_chronic, p_normal, p_vp]

        return dl_pred, final_pred, dl_conf, source, meta, probs
    
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
    y_true, y_pred, y_score = [], [], []
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
            dl_pred, final_pred, dl_conf, source, meta, probs = classifier.predict(crop_img, vcr_ratio)

            # Save Result
            gt_key = (str(reg_id), vb_name.lower())
            gt_label = gt_map.get(gt_key, "Unknown")
            
            if gt_label in TARGET_CLASSES:
                y_true.append(gt_label)
                y_pred.append(final_pred)
                y_score.append(probs)
            
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
            
    pd.DataFrame(flat_result).to_excel(output_folder / 'final_result_ensemble.xlsx', index=False)

    return flat_result, y_true, y_pred, y_score

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
    # stage1_dir = "result_weights/P2_imagesize/STAGE1_convnext_base.fb_in22k_ft_in1k" 
    stage1_dir = "result_weights/patient_kfold/STAGE1_convnext_base.fb_in22k_ft_in1k_384" 
    stage2_dir = "result_weights/P6_augmentation/STAGE2_convnext_base.fb_in22k_ft_in1k_384"
    
    n_folds = 5
    
    # 통합 결과를 담을 리스트
    all_flat_results = []
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for fold in range(n_folds):
        output_folder = Path(f"Result_OOF_VAL_FOLD{fold}_0306")
        output_folder.mkdir(parents=True, exist_ok=True)

        regid_set = get_val_regids_for_fold(csv_path, fold)
        print(f"\n======================================")
        print(f"[Fold {fold}] val patients = {len(regid_set)}")
        print(f"======================================")

        image_list = build_image_list_for_fold(input_folder, regid_set)
        
        # 핵심: SingleFoldClassifier 호출, 파라미터 전달
        classifier = SingleFoldClassifier(
            stage1_dir, stage2_dir, fold_idx=fold, 
        )

        flat_result, y_true, y_pred, y_score = inference(image_list, output_folder, classifier, gt_path=Path(csv_path))
        
        all_flat_results.extend(flat_result)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_score.extend(y_score)
        

    pd.DataFrame(all_flat_results).to_excel('final_result_OOF_ALL.xlsx', index=False)
    
    TARGET_CLASSES = ['Acute', 'Chronic', 'Normal', 'VP']
    
    if len(all_y_true) > 0:
        cm = confusion_matrix(all_y_true, all_y_pred, labels=TARGET_CLASSES)
        print("\n" + "="*60)
        print("          🏥 TOTAL OOF MEDICAL METRICS REPORT 🏥")
        print("="*60)
        print("\n[Total OOF Confusion Matrix]")
        print(cm)
        
        def get_binary_metrics(cm, idx):
            tp = cm[idx, idx]
            fn = np.sum(cm[idx, :]) - tp
            fp = np.sum(cm[:, idx]) - tp
            tn = np.sum(cm) - (tp + fn + fp)
            sens = tp / (tp + fn + 1e-6)
            spec = tn / (tn + fp + 1e-6)
            return sens, spec, tp, fn, tn, fp

        print(f"\n{'Class':<10} | {'Sensitivity (민감도)':<20} | {'Specificity (특이도)':<20}")
        print("-" * 60)
        
        for idx, class_name in enumerate(TARGET_CLASSES):
            sens, spec, tp, fn, tn, fp = get_binary_metrics(cm, idx)
            print(f"{class_name:<10} | {sens:.2%} ({tp}/{tp+fn})   | {spec:.2%} ({tn}/{tn+fp})")
        print("-" * 60)

        # 통합 골절 vs 정상
        total_normal = np.sum(cm[2, :])
        correct_normal = cm[2, 2]
        total_fracture = np.sum(cm) - total_normal
        
        correct_fracture_detection = 0
        fracture_indices = [0, 1, 3]
        for r in fracture_indices:
            for c in fracture_indices:
                correct_fracture_detection += cm[r, c]

        sys_sens = correct_fracture_detection / (total_fracture + 1e-6)
        sys_spec = correct_normal / (total_normal + 1e-6)

        print("\n[📢 EXECUTIVE SUMMARY (골절 vs 정상)]")
        print(f"1. Total Fracture Sensitivity : {sys_sens:.2%}")
        print(f"2. Total Normal Specificity   : {sys_spec:.2%}")
        print("="*60)

    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # y_true 글자(['Acute', 'Normal'...])를 숫자 인덱스로 변환
    label_to_idx = {name: idx for idx, name in enumerate(TARGET_CLASSES)}
    y_true_idx = [label_to_idx[label] for label in all_y_true]
    
    y_true_bin = label_binarize(y_true_idx, classes=[0, 1, 2, 3])
    y_score_np = np.array(all_y_score)

    plt.figure(figsize=(10, 8))
    for i in range(len(TARGET_CLASSES)):
        # fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score_np[:, i])
        # roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=2, label=f'{TARGET_CLASSES[i]} (AUC = {roc_auc:.3f})')

    
        # 💡 Chronic(인덱스 1번)의 최적 Threshold 찾기 (Youden's J)
        fpr, tpr, thresholds_c = roc_curve(y_true_bin[:, i], y_score_np[:, i])
        
        # J = TPR - FPR 계산
        J_scores = tpr - fpr
        best_idx = np.argmax(J_scores) # J값이 가장 큰(최적의) 인덱스 찾기
        best_chronic_threshold = thresholds_c[best_idx]
        
        print(f"[AI 추천] {TARGET_CLASSES[i]:<7} 최적 Threshold: {best_chronic_threshold:.4f}")

        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{TARGET_CLASSES[i]} (AUC = {roc_auc:.3f})')
        plt.scatter(fpr[best_idx], tpr[best_idx], marker='o', color='black', zorder=5)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Multi-Class ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # 서버 저장
    save_path = output_folder / 'Total_OOF_ROC_Curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight') 
    plt.close()