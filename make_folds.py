# make_folds.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

INPUT_EXCEL = "/workspace/snubh/cropping_vertebra_result_from_json_lat_train/crop_summary_from_json_lat.xlsx"
OUTPUT_CSV = "crop_summary_with_folds.csv"
N_SPLITS = 5
SEED = 42

def build_stage1_label(diag: str) -> str:
    d = str(diag).strip()
    if d == "VP":
        return "VP"
    if d in ["Acute", "Chronic"]:
        return "Fracture"
    return "Normal"

def make_patient_folds(df):
    df = df.copy()
    df["RegID"] = df["RegID"].astype(str).str.strip()
    df["Diagnosis"] = df["Diagnosis"].astype(str).str.strip()
    df["s1_label"] = df["Diagnosis"].map(build_stage1_label)

    priority = {"Normal": 0, "Fracture": 1, "VP": 2}

    patient_df = (
        df.groupby("RegID")["s1_label"]
        .apply(lambda s: max(s.map(priority)))
        .reset_index()
        .rename(columns={"s1_label": "priority"})
    )

    inv = {0: "Normal", 1: "Fracture", 2: "VP"}
    patient_df["patient_label"] = patient_df["priority"].map(inv)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    patient_df["fold"] = -1
    X = patient_df["RegID"].values
    y = patient_df["patient_label"].values

    for fold, (_, va_idx) in enumerate(skf.split(X, y)):
        patient_df.loc[va_idx, "fold"] = fold

    df = df.merge(patient_df[["RegID", "fold"]], on="RegID", how="left")
    return df

if __name__ == "__main__":
    df = pd.read_excel(INPUT_EXCEL)
    df = make_patient_folds(df)

    print("Fold distribution:")
    print(df["fold"].value_counts().sort_index())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}")

# def make_patient_folds(
#     df: pd.DataFrame,
#     patient_col: str = "RegID",
#     diag_col: str = "Diagnosis",
#     n_splits: int = 5,
#     seed: int = 42,
# ) -> pd.DataFrame:
#     df = df.copy()
#     df[patient_col] = df[patient_col].astype(str).str.strip()
#     df[diag_col] = df[diag_col].astype(str).str.strip()

#     # row-level stage1 label
#     df["s1_label"] = df[diag_col].map(build_stage1_label)

#     # patient-level 대표 라벨: VP > Fracture > Normal
#     priority = {"Normal": 0, "Fracture": 1, "VP": 2}

#     patient_df = (
#         df.groupby(patient_col)["s1_label"]
#         .apply(lambda s: int(max(s.map(priority))))
#         .reset_index()
#         .rename(columns={"s1_label": "patient_priority"})
#     )
#     inv = {0: "Normal", 1: "Fracture", 2: "VP"}
#     patient_df["patient_label"] = patient_df["patient_priority"].map(inv)

#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

#     patient_df["fold"] = -1
#     X = patient_df[patient_col].values
#     y = patient_df["patient_label"].values

#     for fold, (_, va_idx) in enumerate(skf.split(X, y)):
#         patient_df.loc[va_idx, "fold"] = fold

#     # merge fold back
#     df = df.merge(patient_df[[patient_col, "fold", "patient_label"]], on=patient_col, how="left")
#     if (df["fold"] < 0).any():
#         raise RuntimeError("Some rows did not get a fold assigned. Check patient_col consistency.")

#     return df


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--in_excel", type=str, required=True, help="Input excel path")
#     parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path with fold column")
#     parser.add_argument("--sheet", type=str, default=None, help="Excel sheet name (optional)")
#     parser.add_argument("--patient_col", type=str, default="RegID")
#     parser.add_argument("--diag_col", type=str, default="Diagnosis")
#     parser.add_argument("--n_splits", type=int, default=5)
#     parser.add_argument("--seed", type=int, default=42)
#     args = parser.parse_args()

#     df = pd.read_excel(args.in_excel, sheet_name=args.sheet)
#     df_folds = make_patient_folds(
#         df,
#         patient_col=args.patient_col,
#         diag_col=args.diag_col,
#         n_splits=args.n_splits,
#         seed=args.seed,
#     )

#     # sanity: patient leakage check (same patient in multiple folds)
#     leakage = (
#         df_folds.groupby(args.patient_col)["fold"]
#         .nunique()
#         .sort_values(ascending=False)
#     )
#     if leakage.max() != 1:
#         bad = leakage[leakage > 1]
#         raise RuntimeError(f"Patient leakage detected! Patients in multiple folds: {bad.index[:10].tolist()}")

#     print("[INFO] Fold distribution (patient_label by fold):")
#     print(pd.crosstab(df_folds["fold"], df_folds["patient_label"]))

#     df_folds.to_csv(args.out_csv, index=False)
#     print(f"[DONE] Saved: {args.out_csv}")


# if __name__ == "__main__":
#     main()