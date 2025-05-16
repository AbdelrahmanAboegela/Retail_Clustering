import json
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

# Define file paths
RAW_JSON = "data/raw/transactions.json"
PROC_CSV = "data/processed/clean_transactions.csv"
RFM_CSV  = "data/processed/rfm.csv"

def clean_transaction_data(df):
    """
    Clean and preprocess transaction data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw transaction data
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned transaction data
    """
    # Remove negative quantities and missing customer IDs
    cleaned_df = df[df["Quantity"] > 0].dropna(subset=["Customer ID"])
    
    # Convert invoice date to datetime
    cleaned_df["InvoiceDate"] = pd.to_datetime(cleaned_df["InvoiceDate"])
    
    return cleaned_df

def create_rfm_features(transaction_df):
    """
    Create RFM (Recency, Frequency, Monetary) features from transaction data
    
    Parameters:
    -----------
    transaction_df : pandas.DataFrame
        Cleaned transaction data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RFM features per customer
    """
    # Define snapshot date (day after the last transaction)
    snapshot_date = transaction_df["InvoiceDate"].max() + timedelta(days=1)
    
    # Aggregate by customer ID to compute RFM metrics
    rfm_data = transaction_df.groupby("Customer ID").agg({
        "InvoiceDate": lambda x: (snapshot_date - x.max()).days,   # Recency - days since last purchase
        "Invoice"    : "nunique",                                  # Frequency - number of transactions
        "TotalPrice" : "sum"                                       # Monetary - total spend amount
    }).rename(columns={
        "InvoiceDate": "Recency",
        "Invoice": "Frequency",
        "TotalPrice": "Monetary"
    })
    
    return rfm_data

def scale_rfm_data(rfm_data):
    """
    Standardize RFM features for clustering
    
    Parameters:
    -----------
    rfm_data : pandas.DataFrame
        RFM data
    
    Returns:
    --------
    pandas.DataFrame
        Scaled RFM data
    """
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Scale the data and create a new DataFrame
    scaled_data = pd.DataFrame(
        scaler.fit_transform(rfm_data),
        index=rfm_data.index,
        columns=rfm_data.columns
    )
    
    return scaled_data

def main():
    """Main function to process the data"""
    # Create directories if they don't exist
    os.makedirs("data/processed", exist_ok=True)
    
    # Load raw transaction data
    print("Loading transaction data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw_data = pd.DataFrame(json.load(f)["records"])

    # Clean data
    print("Cleaning transaction data...")
    clean_data = clean_transaction_data(raw_data)
    clean_data.to_csv(PROC_CSV, index=False)

    # Generate RFM features
    print("Creating RFM features...")
    rfm_data = create_rfm_features(clean_data)
    
    # Scale RFM data for clustering
    print("Scaling RFM features...")
    scaled_rfm_data = scale_rfm_data(rfm_data)
    scaled_rfm_data.to_csv(RFM_CSV)

    # Print summary
    print(f"\nData processing completed:")
    print(f"[transform] Cleaned transactions saved to: {PROC_CSV}")
    print(f"[transform] RFM data (scaled) saved to: {RFM_CSV}")
    print(f"[transform] Number of customers: {len(rfm_data)}")
    
    # Print RFM summary statistics
    print("\nRFM Summary Statistics:")
    print(rfm_data.describe().round(2))

if __name__ == "__main__":
    main()
