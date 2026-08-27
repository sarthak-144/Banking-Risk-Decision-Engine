import os
import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, abs

def upload_directory_to_s3(bucket_name, local_dir, s3_prefix):
    """
    Parquet isn't a single file; it's a directory of partitioned files.
    This function walks the directory and uploads all parts to S3 using boto3.
    """
    s3_client = boto3.client('s3', region_name='ap-south-1') 
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            
            # Construct the S3 path maintaining the folder structure
            relative_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(s3_prefix, relative_path)
            
            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_path}...")
            s3_client.upload_file(local_path, bucket_name, s3_path)

def process_data_locally_and_upload(bucket_name):
    # 1. Initialize Spark WITH INCREASED MEMORY
    print("Initializing Spark Session with 4GB RAM allocation...")
    spark = SparkSession.builder \
        .appName("PaySim_Data_Processing") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    # 2. Read the local CSV we downloaded in step 1
    local_csv_path = "dataset/PS_20174392719_1491204439457_log.csv" 
    print(f"Reading local data from {local_csv_path}...")
    df = spark.read.csv(local_csv_path, header=True, inferSchema=True)
    
    # 3. Feature Engineering Layer
    print("Engineering risk and commercial features...")
    processed_df = df.withColumn(
        "orig_balance_discrepancy",
        abs((col("oldbalanceOrg") - col("amount")) - col("newbalanceOrig"))
    ).withColumn(
        "dest_balance_discrepancy",
        abs((col("oldbalanceDest") + col("amount")) - col("newbalanceDest"))
    )

    processed_df = processed_df.withColumn(
        "is_massive_transfer",
        when(col("amount") > 200000, 1).otherwise(0)
    )

    print("Sample of Transformed Data:")
    processed_df.select("step", "type", "amount", "orig_balance_discrepancy", "is_massive_transfer", "isFraud").show(5)

    # 4. Save locally as Parquet in partitions
    local_parquet_dir = "processed_paysim.parquet"
    print(f"Saving optimized Parquet data locally to {local_parquet_dir}...")
    
    # Repartitioning into 8 chunks prevents memory overload during the write phase
    processed_df.repartition(8).write.mode("overwrite").parquet(local_parquet_dir)
    print("Spark processing complete!")
    
    spark.stop()

    # 5. Push to S3 Data Lake
    print("\nPushing Parquet partitions to AWS S3...")
    upload_directory_to_s3(bucket_name, local_parquet_dir, "processed/paysim_features.parquet")
    print("Pipeline Complete! Data is ready in your Data Lake.")

if __name__ == "__main__":
    my_unique_bucket = "paysim-risk-engine-sjain-2026"
    process_data_locally_and_upload(my_unique_bucket)