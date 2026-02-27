import re
import pandas as pd

gt_csv = "Crescom_comp_Fx (최종송부용)_edited.csv"
gt_df = pd.read_csv(gt_csv)

def normalize_gt_cell(cell: str) -> str:
    if pd.isna(cell):
        return ""
    s = str(cell)

    # 줄바꿈/세미콜론 등 통일
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace(";", " ").replace("/", " ")
    s = re.sub(r"\s+", " ", s).strip()

    # 핵심: "L2, 3" / "T12, 11, 10" 같은 형태를 "L2, L3" / "T12, T11, T10"로
    # 패턴: ([TL])(\d+) , (숫자들...)
    def repl(m):
        prefix = m.group(1).upper()
        first = int(m.group(2))
        rest = [int(x) for x in re.findall(r"\d{1,2}", m.group(3))]
        # 예: "L2, 3, 4" -> "L2, L3, L4"
        parts = [f"{prefix}{first}"] + [f"{prefix}{n}" for n in rest]
        return ", ".join(parts)

    # "L2, 3" 처럼 prefix 없는 숫자들이 뒤에 붙은 케이스만 변환
    s = re.sub(r"\b([TtLl])\s*(\d{1,2})\s*,\s*((?:\d{1,2}\s*,\s*)*\d{1,2})\b", repl, s)

    return s


def parse_vb_list(cell) -> list[str]:
    """
    Examples:
      "T11,12,L1,2,3,4,5" -> ["T11","T12","L1","L2","L3","L4","L5"]
      "L2, 3" -> ["L2","L3"]
      "T12\nL2,3" -> ["T12","L2","L3"]
      "L2, L3" -> ["L2","L3"]
    """
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s == "":
        return []

    # normalize separators
    s = s.replace("\n", ",").replace("\r", ",")
    s = s.replace(";", ",").replace("/", ",")
    s = re.sub(r"\s+", "", s)  # remove all whitespace

    # split by comma
    toks = [t for t in s.split(",") if t != ""]

    out = []
    cur_prefix = None  # "T" or "L"
    for tok in toks:
        # token like "T11" or "l3"
        m = re.fullmatch(r"([TtLl])(\d{1,2})", tok)
        if m:
            cur_prefix = m.group(1).upper()
            out.append(f"{cur_prefix}{int(m.group(2))}")
            continue

        # token like "12" (needs prefix)
        m = re.fullmatch(r"(\d{1,2})", tok)
        if m and cur_prefix is not None:
            out.append(f"{cur_prefix}{int(m.group(1))}")
            continue

        # token could be weird; ignore but you can log if needed
        # print("Unparsed vb token:", tok)

    # de-dup while keeping order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


long_rows = []
list_rows = []

COL_REGID = "RegID"
COL_ACT = "Acute"
COL_CHR = "Chronic"
COL_VP = "VP"

for _, r in gt_df.iterrows():
    regid = r[COL_REGID]
    act_cell = normalize_gt_cell(r.get(COL_ACT, ""))
    chr_cell = normalize_gt_cell(r.get(COL_CHR, ""))
    vp_cell = normalize_gt_cell(r.get(COL_VP, ""))
    
    act_vb = parse_vb_list(act_cell)
    chr_vb = parse_vb_list(chr_cell)
    vp_vb = parse_vb_list(vp_cell)
    
    for vb in vp_vb:
        long_rows.append({"RegID": regid, "vb": vb, "stage": "VP", "gt_fracture": 1})    
    for vb in act_vb:
        long_rows.append({"RegID": regid, "vb": vb, "stage": "Acute", "gt_fracture": 1})
    for vb in chr_vb:
        long_rows.append({"RegID": regid, "vb": vb, "stage": "Chronic", "gt_fracture": 1})
        
    merged = []
    seen = set()
    for vb in act_vb + chr_vb + vp_vb:
        if vb not in seen:
            merged.append(vb)
            seen.add(vb)
            
    list_rows.append({
        "RegID": regid,
        "gt_vb_list": ",".join(merged),
        "gt_vb_count": len(merged),
        "gt_VP_list": ",".join(vp_vb),
        "gt_acute_list": ",".join(act_vb),
        "gt_chronic_list": ",".join(chr_vb)
    })
    
    
gt_long = pd.DataFrame(long_rows).sort_values(["RegID", "vb", "stage"])
gt_list = pd.DataFrame(list_rows).sort_values(["RegID"])

gt_long.to_excel('gt_long.xlsx', index=False)
gt_list.to_excel('gt_list.xlsx', index=False)