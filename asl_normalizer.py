"""
Normalize asl_landmarks_final.csv for training.

INPUT COLUMN LAYOUT (interleaved, as provided):
    x0,y0,z0, x1,y1,z1, ..., x20,y20,z20, label

WHAT THIS SCRIPT DOES:
1. Drops rows labeled 'J' and 'Z' (motion letters — can't be represented by
   a single static frame, so they don't belong in a static-fingerspelling
   classifier), and 'del'/'space' (control tokens, not ASL letters — out of
   scope for a letter-recognition classifier).
2. Translation: subtracts the wrist (landmark 0: x0,y0,z0) from every other
   landmark in the row, so hand position within the camera frame becomes
   irrelevant.
3. Scale: divides every coordinate by the max Euclidean distance from the
   wrist to any other landmark in that row, so distance from the camera
   becomes irrelevant.

INPUT:  asl_landmarks_final.csv
OUTPUT: normalized_dataset.csv (same column layout, normalized values,
        J/Z rows removed)
"""

import pandas as pd
import numpy as np

INPUT_CSV = "asl_landmarks_final.csv"
OUTPUT_CSV = "normalized_dataset.csv"
NUM_LANDMARKS = 21
DROP_LABELS = {"J", "Z", "del", "space"}

df = pd.read_csv(INPUT_CSV)

print(f"Loaded {len(df)} rows.")
print(f"Label counts before drop:\n{df['label'].value_counts().sort_index()}")

# 1. Drop motion-letter rows
df = df[~df["label"].isin(DROP_LABELS)].reset_index(drop=True)
print(f"\nRows remaining after dropping J/Z/del/space: {len(df)}")

# Column names in interleaved order: x0,y0,z0,x1,y1,z1,...,x20,y20,z20
coord_cols = []
for i in range(NUM_LANDMARKS):
    coord_cols += [f"x{i}", f"y{i}", f"z{i}"]

normalized_rows = []

for _, row in df.iterrows():
    coords = row[coord_cols].values.astype(float).reshape(NUM_LANDMARKS, 3)

    # 2. Translate: wrist (landmark 0) becomes the origin
    wrist = coords[0].copy()
    translated = coords - wrist

    # 3. Scale: divide by max distance from wrist to any other landmark
    distances = np.linalg.norm(translated, axis=1)
    max_distance = distances.max()
    if max_distance == 0:
        continue  # degenerate row (all landmarks collapsed to a point), skip

    normalized = translated / max_distance

    new_row = {}
    for i in range(NUM_LANDMARKS):
        new_row[f"x{i}"] = normalized[i, 0]
        new_row[f"y{i}"] = normalized[i, 1]
        new_row[f"z{i}"] = normalized[i, 2]
    new_row["label"] = row["label"]
    normalized_rows.append(new_row)

out_df = pd.DataFrame(normalized_rows, columns=coord_cols + ["label"])
out_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(out_df)} normalized rows -> {OUTPUT_CSV}")
print(
    f"\nLabel counts after normalization:\n{out_df['label'].value_counts().sort_index()}"
)

# Sanity check: wrist columns should now be ~0 for every row
print(
    f"\nSanity check (should be ~0): x0 mean={out_df['x0'].mean():.6f}, "
    f"y0 mean={out_df['y0'].mean():.6f}, z0 mean={out_df['z0'].mean():.6f}"
)
