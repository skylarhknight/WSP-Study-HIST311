import requests
import pandas as pd
import wbdata
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --- CONFIGURATION ---
# WHO Indicator: Prevalence of insufficient physical activity (Age-standardized, 18+)
WHO_INDICATOR = "NCD_PAC_ADULTS" 
# World Bank Indicator: Births attended by skilled health staff (% of total)
# (Proxy for women's access to professional healthcare)
WB_INDICATOR = "SH.STA.BRTC.ZS" 
TARGET_YEAR = 2016  # WHO PA data is often sporadic; 2016 is a major data year for this specific metric

def get_who_data(indicator_code, year):
    """
    Fetches specific indicator data from WHO OData API for female population.
    """
    print(f"Fetching WHO data for indicator: {indicator_code}...")
    # OData filter: Indicator code AND Year AND Sex (FMLE)
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    
    # We fetch all and filter in pandas for simplicity, or use OData filters
    response = requests.get(url)
    data = response.json()['value']
    
    df = pd.DataFrame(data)
    
    # Filter for relevant columns and values
    # Dim1 often holds 'SEX'. We need 'FMLE'
    # SpatialDim is the country code (ISO3)
    # TimeDim is the year
    
    # Note: WHO API structures can vary; we check for 'Dim1' (Sex)
    if 'Dim1' in df.columns:
        df = df[df['Dim1'] == 'FMLE']
    
    df = df[df['TimeDim'] == year]
    
    # Extract Country Code and Value
    # WHO uses numeric string in 'NumericValue' usually
    df = df[['SpatialDim', 'NumericValue']]
    df.columns = ['country_code', 'insufficient_activity_female']
    
    # Drop NaNs
    df.dropna(inplace=True)
    return df

def get_world_bank_data(indicator_code, year):
    """
    Fetches data from World Bank API using wbdata library.
    """
    print(f"Fetching World Bank data for indicator: {indicator_code}...")
    
    # wbdata takes a date object or string
    data_date = (pd.to_datetime(f"{year}-01-01"), pd.to_datetime(f"{year}-01-01"))
    
    # Fetch data
    df = wbdata.get_dataframe({indicator_code: 'healthcare_access'}, country='all', data_date=data_date)
    
    # The index is 'country', we need ISO3 codes. 
    # wbdata returns country names as index usually, let's reset to get codes if possible
    # A cleaner way with wbdata to ensure codes is to fetch the series and map IDs
    
    # Alternative robust fetch:
    indicators = {indicator_code: 'healthcare_access'}
    df = wbdata.get_dataframe(indicators, data_date=data_date)
    
    # Reset index to get Country names, but we need codes to match WHO.
    # We will use a helper to map names to codes or fetch with ID included.
    # Let's use standard requests for WB to ensure we get ISO3 codes easily if wbdata is tricky with IDs
    # actually, wbdata index is usually Country Name. We can map it, but let's stick to a robust method:
    
    countries = wbdata.get_country()
    country_map = {c['name']: c['id'] for c in countries}
    
    df.reset_index(inplace=True)
    df['country_code'] = df['country'].map(country_map)
    
    return df[['country_code', 'healthcare_access']].dropna()

def analyze_and_plot(df):
    """
    Performs statistical analysis and generates plots.
    """
    print(f"Analyzing {len(df)} countries...")
    
    # 1. Statistical Correlation
    corr, p_value = stats.pearsonr(df['healthcare_access'], df['insufficient_activity_female'])
    print(f"\n--- Results ---")
    print(f"Pearson Correlation: {corr:.4f}")
    print(f"P-value: {p_value:.4e}")
    
    if p_value < 0.05:
        print("Result: Statistically Significant correlation found.")
    else:
        print("Result: No statistically significant correlation.")

    # 2. Visualization
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot with regression line
    sns.regplot(x='healthcare_access', y='insufficient_activity_female', data=df, 
                scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    plt.title(f"Women's Healthcare Access vs. Physical Inactivity ({TARGET_YEAR})", fontsize=15)
    plt.xlabel("Births Attended by Skilled Staff (% of total) \n(Proxy for Healthcare Access)", fontsize=12)
    plt.ylabel("Insufficient Physical Activity (Female %) \n(Lower is Better)", fontsize=12)
    
    # Add text box with stats
    text_str = f'Correlation: {corr:.2f}\nP-value: {p_value:.1e}'
    plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def main():
    try:
        # 1. Get WHO Data
        who_df = get_who_data(WHO_INDICATOR, TARGET_YEAR)
        print(f"Retrieved WHO data for {len(who_df)} countries.")
        
        # 2. Get World Bank Data
        wb_df = get_world_bank_data(WB_INDICATOR, TARGET_YEAR)
        print(f"Retrieved World Bank data for {len(wb_df)} countries.")
        
        # 3. Merge Datasets on Country Code
        merged_df = pd.merge(who_df, wb_df, on='country_code', how='inner')
        
        if merged_df.empty:
            print("Error: No overlapping countries found. Check year or country codes.")
            return

        # 4. Analyze
        analyze_and_plot(merged_df)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()