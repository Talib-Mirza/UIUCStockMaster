from preprocessing_data import load_datasets, standardize_dates, align_dates_with_ohlcv, merge_datasets
import pandas as pd

def create_df(limit=["technical_data", "economic_data", "financial", "cashflow", "balance_sheet"]):
    valid_values = {
        "technical_data",
        "economic_data",
        "financial",
        "cashflow",
        "balance_sheet"
    }
    
    invalid = [item for item in limit if item not in valid_values]
    
    if invalid:
        raise ValueError(f"Invalid dataset: {', '.join(invalid)}")

    datasets = load_datasets()
    
    # Standardize date formats
    datasets = standardize_dates(datasets)
    
    datasets = align_dates_with_ohlcv(datasets)
    # Merge datasets
    final_df = merge_datasets(datasets, limit=limit)

    # Filter by date range and reset index to make date a column
    final_df = final_df[(final_df.index >= '2019-01-01') & (final_df.index <= '2025-03-28')]
    final_df = final_df.reset_index()
    final_df = final_df.rename(columns={'index': 'Date'})
    
    return final_df
    
