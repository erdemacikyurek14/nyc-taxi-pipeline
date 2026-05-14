import glob, pandas as pd
files = glob.glob('/app/data/delta/gold/fare_features/*.parquet')
print("Files:", len(files))
if files:
    df = pd.read_parquet(files[0])
    print("Columns:", list(df.columns))
    col = 'pickup_day_of_week'
    if col in df.columns:
        print(col, "unique:", sorted(df[col].dropna().unique().tolist()))
