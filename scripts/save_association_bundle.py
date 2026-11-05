"""
Save the association rules from the Association Rule Mining lab as a joblib
bundle for the API, in the same style as the other members' models.

The rules are NOT re-mined. They are loaded exactly as saved in
AssociationRuleMining-C1/outputs/ and written to
models/association_rules_bundle.joblib.

Usage (from the repo root, with the .venv active):
    python scripts/save_association_bundle.py <path-to-AssociationRuleMining-C1/outputs>
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

outputs = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../AssociationRuleMining-C1/outputs")
out_file = Path(__file__).resolve().parent.parent / "models" / "association_rules_bundle.joblib"

rules = pd.read_pickle(outputs / "clean_association_rules.pkl")
lookup = pd.read_csv(outputs / "show_lookup.csv")

bundle = {
    # Rules stored as plain Python (frozensets, floats) so loading them
    # does not depend on the pandas version installed on the server.
    "rules": [
        {
            "antecedents": frozenset(r.antecedents),
            "consequents": frozenset(r.consequents),
            "support": float(r.support),
            "confidence": float(r.confidence),
            "lift": float(r.lift),
        }
        for r in rules.itertuples()
    ],
    "show_names": {str(k): str(v) for k, v in zip(lookup["show_id"], lookup["item_name"])},
    "thresholds": {"min_support": 0.02, "min_confidence": 0.30, "min_lift": 2.0},
}

out_file.parent.mkdir(exist_ok=True)
joblib.dump(bundle, out_file)
print(f"Saved {len(bundle['rules'])} rules and {len(bundle['show_names'])} shows to {out_file}")
