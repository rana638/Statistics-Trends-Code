# GDP_analysis.py
# Created for Statistics & Trends Module, 2025
# Analysis of World Bank GDP per capita (NY.GDP.PCAP.CD)
# Author: R
#
# Usage:
# 1. Place the World Bank CSV file (API_NY.GDP.PCAP.CD_...csv) in the 'data/' folder.
# 2. Install requirements: pip install -r requirements.txt
# 3. Run: python GDP_analysis.py
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from pathlib import Path

sns.set(style='whitegrid')
BASE = Path(__file__).parent
DATA_FILE = BASE / 'data' / 'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_134819.csv'
META_FILE = BASE / 'data' / 'Metadata_Country_API_NY.GDP.PCAP.CD_DS2_en_csv_v2_134819.csv'
OUT = BASE / 'outputs'
OUT.mkdir(exist_ok=True)

def load_and_clean(path=DATA_FILE, meta_path=META_FILE):
    df = pd.read_csv(path, skiprows=4)
    year_cols = [c for c in df.columns if c.isdigit()]
    df_long = df.melt(id_vars=['Country Name','Country Code'], value_vars=year_cols,
                      var_name='Year', value_name='GDP_per_capita')
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce').astype('Int64')
    df_long['GDP_per_capita'] = pd.to_numeric(df_long['GDP_per_capita'], errors='coerce')
    df_long = df_long.dropna(subset=['GDP_per_capita'])
    # merge metadata if available
    try:
        meta = pd.read_csv(meta_path)
        meta_small = meta[['Country Code','Region','IncomeGroup']].drop_duplicates()
        df_long = df_long.merge(meta_small, on='Country Code', how='left')
    except Exception:
        pass
    return df_long

def compute_moments(df):
    arr = df['GDP_per_capita'].dropna().values
    moments = {
        'mean': float(np.mean(arr)),
        'variance': float(np.var(arr, ddof=1)),
        'skewness': float(skew(arr, bias=False)),
        'kurtosis': float(kurtosis(arr, fisher=True, bias=False))
    }
    return moments

def save_moments(df, out=OUT):
    # overall and latest year
    overall = compute_moments(df)
    latest_year = int(df['Year'].max())
    latest = df[df['Year']==latest_year]
    latest_m = compute_moments(latest)
    out_df = pd.DataFrame([{'variable':'all_obs', **overall},
                           {'variable':f'year_{latest_year}', **latest_m}])
    out_df.to_csv(out / 'moments.csv', index=False)
    return out_df

def plot_relational(df, countries=None, out=OUT):
    if countries is None:
        countries = ['United States','United Kingdom','China','India','Pakistan']
    plt.figure(figsize=(10,6))
    for c in countries:
        s = df[df['Country Name']==c].sort_values('Year')
        if not s.empty:
            plt.plot(s['Year'], s['GDP_per_capita'], marker='o', label=c)
    plt.title('GDP per Capita Trends (1960–2024)')
    plt.xlabel('Year'); plt.ylabel('GDP per capita (current US$)')
    plt.legend(); plt.grid(alpha=0.4)
    plt.tight_layout()
    path = out / 'gdp_lineplot.png'
    plt.savefig(path, dpi=300); plt.close()
    return path

def plot_categorical(df, countries=None, out=OUT):
    latest_year = int(df['Year'].max())
    if countries is None:
        countries = ['United States','United Kingdom','China','India','Pakistan']
    latest = df[(df['Year']==latest_year) & (df['Country Name'].isin(countries))]
    plt.figure(figsize=(8,5))
    sns.barplot(data=latest, x='Country Name', y='GDP_per_capita', palette='viridis')
    plt.title(f'GDP per Capita ({latest_year}) - selected countries')
    plt.ylabel('GDP per capita (current US$)'); plt.xlabel('Country')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    path = out / 'gdp_barplot.png'
    plt.savefig(path, dpi=300); plt.close()
    return path

def plot_box_by_region(df, out=OUT):
    latest_year = int(df['Year'].max())
    latest = df[df['Year']==latest_year].dropna(subset=['Region'])
    plt.figure(figsize=(12,6))
    sns.boxplot(data=latest, x='Region', y='GDP_per_capita')
    plt.title(f'Regional GDP per Capita Distribution ({latest_year})')
    plt.xticks(rotation=45); plt.ylabel('GDP per capita (current US$)')
    plt.tight_layout()
    path = out / 'gdp_boxplot.png'
    plt.savefig(path, dpi=300); plt.close()
    return path

def main():
    df = load_and_clean()
    moments = save_moments(df)
    print('Saved moments:\n', moments)
    p1 = plot_relational(df)
    p2 = plot_categorical(df)
    try:
        p3 = plot_box_by_region(df)
    except Exception as e:
        print('Boxplot skipped:', e)
        p3 = None
    print('Plots saved to outputs folder.')
    return

if __name__ == '__main__':
    main()
