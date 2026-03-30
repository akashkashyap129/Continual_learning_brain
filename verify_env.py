# verify_env.py — run this to confirm your environment is set up correctly

import sys

checks = []

try:
    import numpy as np
    checks.append(("numpy", np.__version__, True))
except ImportError:
    checks.append(("numpy", "MISSING", False))

try:
    import pandas as pd
    checks.append(("pandas", pd.__version__, True))
except ImportError:
    checks.append(("pandas", "MISSING", False))

try:
    import sklearn
    checks.append(("scikit-learn", sklearn.__version__, True))
except ImportError:
    checks.append(("scikit-learn", "MISSING", False))

try:
    import matplotlib
    checks.append(("matplotlib", matplotlib.__version__, True))
except ImportError:
    checks.append(("matplotlib", "MISSING", False))

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(["python machine learning"])
    assert vec.shape == (1, 384)
    checks.append(("sentence-transformers", f"OK — vector shape {vec.shape}", True))
except Exception as e:
    checks.append(("sentence-transformers", f"ERROR: {e}", False))

try:
    import umap
    checks.append(("umap-learn", umap.__version__, True))
except ImportError:
    checks.append(("umap-learn", "MISSING", False))

try:
    import torch
    checks.append(("torch", torch.__version__, True))
except ImportError:
    checks.append(("torch", "MISSING", False))

try:
    import fastapi
    checks.append(("fastapi", fastapi.__version__, True))
except ImportError:
    checks.append(("fastapi", "MISSING", False))

print(f"\nPython: {sys.version}\n")
print(f"{'Package':<25} {'Status':<40} {'OK'}")
print("-" * 72)
for name, version, ok in checks:
    status = "OK" if ok else "FAIL"
    print(f"{name:<25} {str(version):<40} {status}")

all_ok = all(ok for _, _, ok in checks)
print("\n" + ("All checks passed." if all_ok else "Some checks FAILED — re-run the pip install steps above."))