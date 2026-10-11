import time
from pathlib import Path
import pandas as pd
import joblib
import json

root = Path("d:/InsightEngine — E-Commerce Customer Analytics Dashboard DS")
art_dir = root / "models" / "segmentation"

t0 = time.perf_counter()
import sklearn.cluster
print(f"Import sklearn: {(time.perf_counter()-t0)*1000:.2f} ms")

t0 = time.perf_counter()
kmeans = joblib.load(art_dir / "kmeans.pkl")
print(f"joblib load kmeans: {(time.perf_counter()-t0)*1000:.2f} ms")

t0 = time.perf_counter()
scaler = joblib.load(art_dir / "scaler.pkl")
print(f"joblib load scaler: {(time.perf_counter()-t0)*1000:.2f} ms")

t0 = time.perf_counter()
with open(art_dir / "metadata.json", "r") as f:
    meta = json.load(f)
print(f"json load metadata: {(time.perf_counter()-t0)*1000:.2f} ms")

t0 = time.perf_counter()
profiles = pd.read_csv(art_dir / "segment_profiles.csv")
print(f"pd read profiles: {(time.perf_counter()-t0)*1000:.2f} ms")

t0 = time.perf_counter()
segments = pd.read_csv(art_dir / "customer_segments.csv")
print(f"pd read segments: {(time.perf_counter()-t0)*1000:.2f} ms")

# Let's test CSV read of cleaned data with and without date_format
t0 = time.perf_counter()
df1 = pd.read_csv(
    root / "data" / "processed" / "cleaned_ecommerce.csv",
    dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "CustomerID": int, "Country": str},
    parse_dates=["InvoiceDate"]
)
print(f"Cleaned CSV default parse_dates: {(time.perf_counter()-t0)*1000:.2f} ms")

t0 = time.perf_counter()
df2 = pd.read_csv(
    root / "data" / "processed" / "cleaned_ecommerce.csv",
    dtype={"InvoiceNo": str, "StockCode": str, "Description": str, "CustomerID": int, "Country": str},
    date_format="%Y-%m-%d %H:%M:%S",
    parse_dates=["InvoiceDate"]
)
print(f"Cleaned CSV date_format: {(time.perf_counter()-t0)*1000:.2f} ms")
