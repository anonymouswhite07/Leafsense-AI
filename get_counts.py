import os
from collections import Counter

data_dir = "data/train"
counts = Counter()
for d in os.listdir(data_dir):
    p = os.path.join(data_dir, d)
    if os.path.isdir(p):
        counts[d] = len(os.listdir(p))

print(json.dumps(counts, indent=2))
