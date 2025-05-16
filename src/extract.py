import os
import json
import datetime
import pandas as pd

# Define file paths
SRC_CSV = "data/raw/online_retail_II.csv"  # Source CSV file
DST_JSON = "data/raw/transactions.json"    # Destination JSON file for transform.py

def extract_transform_online_retail(source_csv):
    """
    Extract data from Online Retail II CSV and transform it for further processing
    
    Parameters:
    -----------
    source_csv : str
        Path to the source CSV file
    
    Returns:
    --------
    list
        List of dictionaries containing transformed records
    """
    # Load the CSV data
    print(f"Reading data from {source_csv}...")
    df = pd.read_csv(source_csv)
    
    # Basic data validation
    required_cols = ["Invoice", "StockCode", "Description", "Quantity", 
                     "InvoiceDate", "Price", "Customer ID", "Country"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {', '.join(missing_cols)}")
    
    # Compute total price (quantity * unit price)
    df["TotalPrice"] = df["Quantity"] * df["Price"]
    
    # Select and rename columns for consistency
    records = df.rename(columns={
        "Price": "UnitPrice",  # Rename for clarity and consistency
        "Customer ID": "Customer ID"
    })[
        ["Invoice", "StockCode", "Description", "Quantity", 
         "InvoiceDate", "UnitPrice", "Customer ID", "Country", "TotalPrice"]
    ].to_dict(orient="records")
    
    print(f"Processed {len(records):,} transaction records")
    return records

def main():
    """Main function to extract retail data and save as JSON"""
    try:
        # Check if source file exists
        if not os.path.exists(SRC_CSV):
            raise FileNotFoundError(
                f"Source file {SRC_CSV} not found. Please copy the Online Retail II CSV file to this location first."
            )
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(DST_JSON), exist_ok=True)
        
        # Extract and transform data
        records = extract_transform_online_retail(SRC_CSV)
        
        # Save to JSON file
        with open(DST_JSON, "w", encoding="utf-8") as fp:
            json.dump({
                "generator": "Online Retail II CSV Extraction",
                "fetched": datetime.datetime.utcnow().isoformat(),
                "record_count": len(records),
                "records": records
            }, fp, ensure_ascii=False)
        
        print(f"[extract] Successfully saved {len(records):,} records to {DST_JSON}")
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
