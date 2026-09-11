import pandas as pd
import json

# Read parquet files
anom = pd.read_parquet('E:/smart energy monitoring sys/data/synthetic/anomalies_meta.parquet')
print('anomalies_meta shape:', anom.shape)
print('anomalies_meta columns:', anom.columns.tolist())
print(anom.head(3).to_string())
print()

faults = pd.read_parquet('E:/smart energy monitoring sys/data/synthetic/faults_meta.parquet')
print('faults_meta shape:', faults.shape)
print('faults_meta columns:', faults.columns.tolist())
print(faults.head(3).to_string())
print()

# Read CSV header info
with open('E:/smart energy monitoring sys/data/synthetic/pzem_historical.csv') as f:
    header = f.readline()
    print('CSV header:', header.strip())

# Count rows in CSV
df_csv = pd.read_csv('E:/smart energy monitoring sys/data/synthetic/pzem_historical.csv')
print('CSV shape:', df_csv.shape)
print('CSV columns:', df_csv.columns.tolist())
print(df_csv.head(3).to_string())