import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "reformed_data"
OUTPUT_DIR = BASE_DIR / "dataset" / "report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(DATA_DIR / "merged_features.csv")

cols = [
    "daily_rental_count",
    "avg_temp_c",
    "min_temp_c",
    "max_temp_c",
    "precip_duration_hr",
    "daily_precip_mm",
    "avg_wind_speed_m_s",
    "avg_rel_humidity_pct",
    "is_weekend",
    "is_rainy",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
]

corr = df[cols].corr(numeric_only=True)

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("따릉이 수요 예측 변수 상관관계 Heatmap")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ddarungi_heatmap.png", dpi=150)
plt.show()

print(f"히트맵 저장 완료: {OUTPUT_DIR / 'ddarungi_heatmap.png'}")
