import json,sys
sys.path.insert(0,".")
from scripts.sft_v14_report import paired_macro_ci
L=lambda p: json.load(open(p))["rows"]
E="runs/eval/"
sets=[("v6","v6"),("v3","v3"),("W4","v6_W4"),("W5","v6_W5"),("heldout","heldout")]
for n,s in sets:
    a,b,c=L(E+f"sft_v11_{s}_t7.json"),L(E+f"sft_v14_{s}_t7.json"),L(E+f"sft_v15b_{s}_t7.json")
    print(n,"v15b-v11",paired_macro_ci(a,c),"| v15b-v14",paired_macro_ci(b,c),flush=True)
print("heldout parity v15b-v11",paired_macro_ci(L(E+"sft_v11_heldout_t7_parity.json"),L(E+"sft_v15b_heldout_t7.json")))
