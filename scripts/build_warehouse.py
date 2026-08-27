import pandas as pd
from sqlalchemy import create_engine, text
import time

def build_data_warehouse():
    # 1. Connect to the PostgreSQL database
    print("Connecting to PostgreSQL...")
    engine = create_engine('postgresql://sjain:admin123@localhost:5432/risk_db')
    
    # 2. Load the processed Parquet data
    print("Loading PySpark Parquet partitions into memory...")
    df = pd.read_parquet("processed_paysim.parquet")
    
    # 3. Push to PostgreSQL
    print(f"Pushing {len(df)} rows to PostgreSQL. This will take a few minutes...")
    start_time = time.time()
    
    # Chunksize prevents memory overflow during millions of row insertions
    df.to_sql(
        'transactions',
        engine,
        if_exists='replace',
        index=False,
        chunksize=20000)

    print(f"Data successfully ingested in {round((time.time() - start_time)/60, 2)} minutes.")

    # 4. Execute complex SQL to build the Risk and LTV Data Marts
    print("Executing SQL to build Risk and Commercial feature tables...")
    with engine.begin() as conn:
        
        # --- COMMERCIAL DOMAIN: LTV FEATURES ---
        # Aggregating spending behavior to predict future customer value
        conn.execute(text("""
            DROP TABLE IF EXISTS customer_ltv_features;
            CREATE TABLE customer_ltv_features AS
            SELECT 
                "nameOrig" as customer_id,
                COUNT(step) as total_transactions,
                SUM(amount) as total_volume,
                AVG(amount) as avg_transaction_size,
                MAX(amount) as max_transaction_size
            FROM transactions
            GROUP BY "nameOrig";
        """))
        print("- Created 'customer_ltv_features' table.")

        # --- RISK DOMAIN: FRAUD AGGREGATION FEATURES ---
        # Profiling accounts based on discrepancy behaviors and massive transfer counts
        conn.execute(text("""
            DROP TABLE IF EXISTS account_risk_features;
            CREATE TABLE account_risk_features AS
            SELECT 
                "nameOrig" as account_id,
                SUM(is_massive_transfer) as massive_transfer_count,
                AVG(orig_balance_discrepancy) as avg_balance_discrepancy,
                MAX(orig_balance_discrepancy) as max_balance_discrepancy,
                SUM("isFraud") as historical_fraud_flags
            FROM transactions
            GROUP BY "nameOrig";
        """))
        print("- Created 'account_risk_features' table.")
        
    print("Data Warehouse build complete! Features are ready for modeling.")

if __name__ == "__main__":
    build_data_warehouse()