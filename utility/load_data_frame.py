from pyspark.sql import SparkSession

def load_data_frame(file_path):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    return df

    