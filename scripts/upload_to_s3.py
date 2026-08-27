import os
import zipfile
import boto3
from botocore.exceptions import ClientError
import subprocess

def download_kaggle_dataset():
    # Download the dataset using the kaggle CLI tool via subprocess
    print("Downloading PaySim dataset from Kaggle...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "ealaxi/paysim1"], check=True)
    
    # Unzip the downloaded file
    print("Extracting dataset...")
    with zipfile.ZipFile("paysim1.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    # The extracted file name from this specific Kaggle dataset
    return "PS_20174392719_1491204439457_log.csv"

def create_bucket_and_upload(bucket_name, file_name, object_name, region="us-east-1"):
    # Initialize the S3 client using boto3
    # The client automatically picks up the credentials you set with `aws configure`
    s3_client = boto3.client('s3', region_name=region)

    # Attempt to create the bucket
    try:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region}
            )
        print(f"Bucket '{bucket_name}' created successfully.")
    except ClientError as e:
        if e.response['Error']['Code'] in ['BucketAlreadyOwnedByYou', 'BucketAlreadyExists']:
            print(f"Bucket '{bucket_name}' already exists.")
        else:
            print(f"Error creating bucket: {e}")
            return

    # Upload the file to S3
    try:
        print(f"Uploading '{file_name}' to 's3://{bucket_name}/{object_name}'...")
        # upload_file automatically performs multipart uploads for large files
        s3_client.upload_file(file_name, bucket_name, object_name)
        print("Upload complete!")
    except ClientError as e:
        print(f"Error uploading file: {e}")

if __name__ == "__main__":
    # S3 buckets must be globally unique across all AWS users. 
    my_unique_bucket = "paysim-risk-engine-sjain-2026"
    
    csv_filename = "dataset/PS_20174392719_1491204439457_log.csv"
    create_bucket_and_upload(my_unique_bucket, csv_filename, "raw/paysim.csv")