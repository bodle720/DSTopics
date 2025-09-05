# AWS Python Tutorial Suite

This directory contains Python applications leveraging various AWS services. Some serve both as step-by-step tutorials and as frameworks you can adapt for your own needs—swap in your own logic, buckets, and VPC settings in place of the existing examples.

## Projects

1. **EC2_ECR_Example**  
   A comprehensive walkthrough demonstrating how to:
   - Build a Docker image containing a Python worker.  
   - Push that image to Amazon ECR.  
   - Launch EC2 instances with a UserData bootstrap script.  
   - Use AWS Systems Manager (SSM) to pull and run the Docker container on each instance.  
   - Save each worker’s output as a Parquet file under `ec2_results/output_{input_value}.parquet` in S3.  
   - Terminate the EC2 instances automatically once tasks complete.  

   Use this code as a template for running containerized workloads on EC2 without manual SSH or SCP.

2. **S3_Examples**  
   A focused collection of scripts showing how to:
   - Upload and download various object types (JSON, CSV, Parquet, etc.) to and from Amazon S3.  
   - Normalize Python dictionaries into pandas DataFrames.  
   - Write DataFrames to Parquet and store them in S3.  
   - Retrieve objects and load them back into your Python environment.  

   Includes reusable helper functions (`helpers.py`) for common S3 operations.


