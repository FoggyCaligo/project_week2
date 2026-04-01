import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset" / "reformed_data"
OUTPUT_DIR = BASE_DIR / "dataset" / "report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "merged_features.csv")

print(df.info())
print(df.head())

profile = ProfileReport(
    df,
    title="Ddarungi Demand Prediction Profiling Report",
    explorative=True
)

output_file = OUTPUT_DIR / "ddarungi_profiling_report.html"
profile.to_file(output_file)

print(f"프로파일링 보고서 생성 완료: {output_file}")
