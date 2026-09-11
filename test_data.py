import pandas as pd
try:
    df = pd.read_parquet('data/synthetic/pzem_historical.parquet')
    print('Shape:', df.shape)
    print('Columns:', list(df.columns))
    print('First 3 rows:')
    print(df.head(3))
    print()
    if 'pzem_id' in df.columns:
        print('Unique meters:', df['pzem_id'].unique())
    print('Timestamp range:', df['timestamp'].min(), '-', df['timestamp'].max())
    if 'timestamp' in df.columns:
        print('Timestamp diff mode:', df['timestamp'].diff().mode().iloc[0])
except Exception as e:
    print('Error reading parquet:', e)