import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../Data-processed/crop_recommendation.csv")

# Count samples for each crop
crop_counts = df["label"].value_counts()

# Pie chart
plt.figure(figsize=(10, 10))

plt.pie(
    crop_counts.values,
    labels=crop_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Distribution of Crop Classes")

plt.tight_layout()

plt.savefig(
    "graph5_crop_distribution_pie.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()