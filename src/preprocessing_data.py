import pandas as pd
import datetime as dt

def load_datasets():
    """
    Load all datasets from CSV files
    """
    datasets = {
        'ohlcv_data': pd.read_csv('data/raw/KO/ohlcv_data.csv'),
        'balance_sheet': pd.read_csv('data/raw/KO/balance_sheet.csv'),
        'cashflow': pd.read_csv('data/raw/KO/cashflow.csv'),
        'financial': pd.read_csv('data/raw/KO/financial.csv'),
        'economic_data': pd.read_csv('data/raw/economic_data.csv'),
        'technical_data': pd.read_csv('data/raw/KO/technical_data.csv')
    }
    return datasets

def standardize_dates(datasets):
    """
    Standardize date formats across all datasets to MM/DD/YYYY
    """
    for name, df in datasets.items():
        # Get the date column name (usually either 'Date' or first column)
        date_col = 'Date' if 'Date' in df.columns else df.columns[0]
        
        try:
            if name in ['ohlcv_data', 'technical_data']:
                # These already have datetime-like format YYYY-MM-DD
                df[date_col] = pd.Series([pd.to_datetime(date).tz_localize(None) for date in df[date_col]])
            elif name == 'economic_data':
                # Convert to datetime first
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                # For financial statements (already in MM/DD/YYYY)
                df[date_col] = pd.to_datetime(df[date_col])
                
            
            # Set date as index
            df.set_index(date_col, inplace=True)
        except Exception as e:
            print(f"Error processing {name} dataset: {str(e)}")
            print(f"First few rows of {name} date column:")
            print(df[date_col].head())
            raise
    
    return datasets


def align_dates_with_ohlcv(datasets):
    """
    Align dates from other datasets with OHLCV data.
    If a date doesn't exist in OHLCV, increment it until finding a match.
    """
    ohlcv_dates = set(datasets['ohlcv_data'].index)
    
    # Process all datasets except OHLCV and technical data
    datasets_to_align = ['balance_sheet', 'cashflow', 'financial', 'economic_data']
    
    for name in datasets_to_align:
        df = datasets[name]
        new_index = []
        
        for date in df.index:
            current_date = pd.to_datetime(date)
            # Keep incrementing the date until we find a match in OHLCV dates
            while current_date not in ohlcv_dates:
                current_date += pd.Timedelta(days=1)
            new_index.append(current_date)
        
        # Create new DataFrame with aligned dates
        datasets[name] = pd.DataFrame(
            df.values, 
            index=new_index,
            columns=df.columns
        )
    
    return datasets

def merge_datasets(datasets, limit=["technical_data",
        "economic_data",
        "financial",
        "cashflow",
        "balance_sheet"
    ]):
    """
    Merge all datasets on the date index
    """
    # Start with the first dataset
    merged_df = datasets['ohlcv_data']
    dataset = {k: v for k, v in datasets.items() if k in limit}
    for name, df in dataset.items():
        # Merge with outer join to keep all dates
        merged_df = merged_df.join(df, how='left', on='Date')
    
    # Sort index by date
    #merged_df.index = pd.to_datetime(merged_df.index)
    merged_df.sort_index(inplace=True)
    
    # Handle missing values with forward fill followed by backward fill
    merged_df = merged_df.ffill().bfill()

    merged_df = merged_df.loc[:, (merged_df != 0).any(axis=0)]
    
    return merged_df

def main():
    # Load all datasets
    datasets = load_datasets()
    
    # Standardize date formats
    datasets = standardize_dates(datasets)
    

    datasets = align_dates_with_ohlcv(datasets)
    # Merge datasets
    final_df = merge_datasets(datasets)

    final_df = final_df[(final_df.index >= '2019-01-01') & (final_df.index <= '2025-03-28')]
    
    # Save the preprocessed dataset
    final_df.to_csv('data/processed/processed_KO.csv')

if __name__ == "__main__":
    main()
