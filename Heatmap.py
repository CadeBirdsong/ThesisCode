import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER = r"C:\Users\Cade\RealFusionProject"

llm_path   = os.path.join(FOLDER, "all_sentiment_outputs.json")
vader_path = os.path.join(FOLDER, "vader_all_sentiment_outputs.json")

# Reload data
with open(llm_path,   "r", encoding="utf-8") as f:
    llm_data = json.load(f)
with open(vader_path, "r", encoding="utf-8") as f:
    vader_data = json.load(f)

llm_df = pd.DataFrame(llm_data)
vader_df = pd.DataFrame(vader_data)

# filter relevant & numeric
llm_df   = llm_df  [ llm_df ["is_relevant"] & llm_df ["score"].apply(lambda x: isinstance(x, int)) ]
vader_df = vader_df[ vader_df["is_relevant"] & vader_df["score"].apply(lambda x: isinstance(x, int)) ]

# Merge
merged = pd.merge(llm_df[["id", "score"]], vader_df[["id", "score"]],
                  on="id", suffixes=("_llm", "_vader"))

print("Unique LLM scores:", sorted(merged["score_llm"].unique()))
print("Unique VADER scores:", sorted(merged["score_vader"].unique()))

# Build confusion matrix for 1-5
score_range = range(1, 6)
conf = (
    pd.crosstab(merged["score_vader"], merged["score_llm"])
      .reindex(index=score_range, columns=score_range)
      .fillna(0).astype(int)
)

# Identify discrepancies
disc = merged[ merged["score_llm"] != merged["score_vader"] ].copy()
disc["score_sum"] = disc["score_llm"] + disc["score_vader"]

# Sort
disc_sorted = disc.sort_values("score_sum")

# Save CSV
CSV_OUT = os.path.join(FOLDER, "discrepancy_article_ids.csv")
disc_sorted.to_csv(CSV_OUT, index=False)

# Plot heatmap
plt.figure(figsize=(5,4))
plt.imshow(conf, origin="lower", aspect="auto")
plt.title("LLM vs VADER Confusion Matrix (1–5 scale)")
plt.xlabel("LLM Score")
plt.ylabel("VADER Score")
plt.xticks(np.arange(5), range(1,6))
plt.yticks(np.arange(5), range(1,6))
plt.colorbar(label="Count")

for i in range(5):
    for j in range(5):
        v = conf.iat[i, j]
        if v:
            plt.text(j, i, v, ha='center', va='center',
                     fontweight="bold" if i!=j else "normal", fontsize=8)

plt.tight_layout()
plt.show()
