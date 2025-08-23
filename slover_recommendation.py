# -*- coding: utf-8 -*-
"""
Spark-based "When-to-Hail" Recommendation with Neural Network
-----------------------------------------------------------
Integrates data preprocessing and neural network recommendation model, implementing:
1. Preserving original data loading, cleaning and feature engineering
2. GPU-based waiting time prediction model training (neural network)
3. Ride-hailing time recommendation based on predictions
4. Result visualization and model saving
"""

import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import (col, hour, dayofweek, month, when, count, avg,
                                   regexp_extract, create_map, lit, unix_timestamp)


def init_spark_session():
    """Initialize Spark session and configure visualization parameters"""
    spark = SparkSession.builder \
        .appName("NYC FHV Trip Analysis with Neural Network") \
        .config("spark.driver.memory", "8g")  \
        .config("spark.executor.memory", "8g")  \
        .config("spark.sql.debug.maxToStringFields", "1000") \
        .getOrCreate()
    
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    sns.set_style("whitegrid")
    
    return spark


def handle_time_anomalies(df):
    """
    处理时间型特征异常：识别并移除请求时间晚于到场时间的异常记录
    """
    total_records = df.count()
    # 筛选出请求时间 <= 到场时间的记录
    df_valid_time = df.filter(col("request_datetime") <= col("on_scene_datetime"))
    valid_records = df_valid_time.count()
    
    anomaly_count = total_records - valid_records
    anomaly_ratio = anomaly_count / total_records * 100 if total_records > 0 else 0
    
    print(f"时间异常处理：总记录数 {total_records}")
    print(f"请求时间晚于到场时间的异常记录数：{anomaly_count} ({anomaly_ratio:.2f}%)")
    print(f"处理后保留的有效记录数：{valid_records}")
    
    return df_valid_time


def handle_numeric_anomalies(df, numeric_cols):
    """
    IQR
    """
    df_clean = df
    total_records = df_clean.count()
    
    for col_name in numeric_cols:
        quantiles = df_clean.approxQuantile(col_name, [0.25, 0.75], 0.01)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        before_count = df_clean.count()
        
        df_clean = df_clean.filter(
            (col(col_name) >= lower_bound) & (col(col_name) <= upper_bound)
        )
        
        anomaly_count = before_count - df_clean.count()
        anomaly_ratio = anomaly_count / before_count * 100 if before_count > 0 else 0
        
        print(f"{col_name} 异常值处理：")
        print(f"  四分位范围: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  异常值数量: {anomaly_count} ({anomaly_ratio:.2f}%)")
        print(f"  处理后记录数: {df_clean.count()}")
    
    final_ratio = (total_records - df_clean.count()) / total_records * 100 if total_records > 0 else 0
    print(f"\n数值型特征异常处理完成：")
    print(f"总记录数从 {total_records} 减少到 {df_clean.count()}")
    print(f"总异常值比例：{final_ratio:.2f}%")
    
    return df_clean


def load_and_preprocess_data(spark, data_dir, sample_ratio=0.01):
    """Load and preprocess data, return the final encoded DataFrame"""
    parquet_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".parquet")]
    parquet_files.sort()
    print(f"Found {len(parquet_files)} Parquet files")

    sampled_dfs = []
    for file in parquet_files:
        df_month = spark.read.parquet(file)
        original_count = df_month.count()
        df_sampled = df_month.sample(withReplacement=False, fraction=sample_ratio, seed=42)
        sampled_count = df_sampled.count()
        print(f"File: {os.path.basename(file)} - Original: {original_count}, Sampled: {sampled_count}")
        sampled_dfs.append(df_sampled)

    df_combined = sampled_dfs[0]
    for df in sampled_dfs[1:]:
        df_combined = df_combined.unionByName(df)
    print(f"Total sampled rows after merging: {df_combined.count()}")

    # 处理时间异常：请求时间晚于到场时间的记录
    df_combined = handle_time_anomalies(df_combined)

    weather_df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("new_york_weather_2024_1-6.csv")
    print(f"Weather data size: {weather_df.count()}")

    zone_df = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load("taxi_zone_lookup.csv")

    weather_with_time = weather_df \
        .withColumn("weather_timestamp", F.to_timestamp(col("row_index"), "yyyy-MM-dd HH:mm")) \
        .withColumn("weather_date", F.to_date(col("weather_timestamp"))) \
        .withColumn("weather_hour", F.hour(col("weather_timestamp")))
    weather_deduped = weather_with_time.dropDuplicates(["weather_date", "weather_hour"])
    print(f"Weather data size after deduplication: {weather_deduped.count()}")

    df_with_time_cols = df_combined \
        .withColumn("request_date", F.to_date(col("request_datetime"))) \
        .withColumn("request_hour", F.hour(col("request_datetime")))

    df_with_weather = df_with_time_cols.join(
        weather_deduped,
        (df_with_time_cols["request_date"] == weather_deduped["weather_date"]) & 
        (df_with_time_cols["request_hour"] == weather_deduped["weather_hour"]),
        how="left"
    )
    print(f"Rows after joining with weather data: {df_with_weather.count()}")

    df_enriched = df_with_weather \
        .withColumn("request_hour", hour(col("request_datetime"))) \
        .withColumn("weekday", dayofweek(col("request_datetime"))) \
        .withColumn("is_weekend", when(col("weekday").isin(1,7), 1).otherwise(0)) \
        .withColumn("trip_month", month(col("request_datetime")))

    pickup_zone_df = zone_df.withColumnRenamed("LocationID", "PULocationID") \
                            .withColumnRenamed("Zone", "PULocationZone") \
                            .select("PULocationID", "PULocationZone", "Borough") \
                            .withColumnRenamed("Borough", "PUBorough")

    dropoff_zone_df = zone_df.withColumnRenamed("LocationID", "DOLocationID") \
                             .withColumnRenamed("Zone", "DOLocationZone") \
                             .select("DOLocationID", "DOLocationZone", "Borough") \
                             .withColumnRenamed("Borough", "DOBorough")

    df_with_zones = df_enriched.join(pickup_zone_df, on="PULocationID", how="left")
    df_with_zones = df_with_zones.join(dropoff_zone_df, on="DOLocationID", how="left")

    print(f"Unmatched PULocationID count: {df_with_zones.filter(col('PULocationZone').isNull()).select('PULocationID').distinct().count()}")
    print(f"Unmatched DOLocationID count: {df_with_zones.filter(col('DOLocationZone').isNull()).select('DOLocationID').distinct().count()}")

    weather_types = df_with_zones.select("weather").distinct().collect()
    weather_mapping = {row.weather: i+1 for i, row in enumerate(weather_types)}
    
    mapping_pairs = []
    for key, value in weather_mapping.items():
        mapping_pairs.extend([lit(key), lit(value)])

    df_weather_encoded = df_with_zones.withColumn(
        "weather_code",
        when(col("weather").isNull(), 0).otherwise(create_map(*mapping_pairs)[col("weather")])
    )

    weather_numeric_cols = [
        ("temp", "temp_num"),
        ("wind", "wind_num"),
        ("humidity", "humidity_num"),
        ("barometer", "barometer_num"),
        ("visibility", "visibility_num")
    ]
    
    for col_name, new_col in weather_numeric_cols:
        df_weather_encoded = df_weather_encoded.withColumn(
            new_col,
            regexp_extract(col(col_name), "(\d+)", 1).try_cast("integer")
        )

    key_columns = [
        "on_scene_datetime","request_datetime","request_date",
        "PULocationID", "DOLocationID", "PULocationZone", "DOLocationZone",
        "PUBorough","DOBorough", "weather_code", "temp_num", "wind_num",
        "request_hour", "weekday", "is_weekend", "trip_month",
        "humidity_num", "barometer_num", "visibility_num"
    ]

    pre_drop_nulls = df_weather_encoded.count()
    df_final_encoded = df_weather_encoded.dropna(subset=key_columns)
    post_drop_nulls = df_final_encoded.count()
    print(f"Null handling: Original {pre_drop_nulls} rows, After handling {post_drop_nulls} rows, Deleted {pre_drop_nulls - post_drop_nulls} rows")

    df_final_encoded = df_final_encoded.select(key_columns)
    df_final_encoded = df_final_encoded.withColumn(
        "wait_minutes",
        F.round(
            (F.unix_timestamp("on_scene_datetime") - F.unix_timestamp("request_datetime")) / 60.0,
            1
        )
    )
    
    numeric_features = [
        "weather_code", "temp_num", "wind_num", 
        "humidity_num", "barometer_num", "visibility_num",
        "request_hour", "weekday", "trip_month",
        "wait_minutes"
    ]
    
    df_final_encoded = handle_numeric_anomalies(df_final_encoded, numeric_features)
    
    return df_final_encoded, weather_mapping


def append_in_batches(df, output_path, batch_size=100):
    """Append write Parquet files in batches of specified row count"""
    total_rows = df.count()
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"Total data volume: {total_rows} rows, will be written in {num_batches} batches, {batch_size} rows per batch")

    df_with_index = df.withColumn("temp_index", F.monotonically_increasing_id())
    
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        
        batch_df = df_with_index.filter(
            (col("temp_index") >= start) & (col("temp_index") < end)
        ).drop("temp_index")
        
        batch_df.write.parquet(output_path, mode="append")
        print(f"Batch {i+1}/{num_batches} written, current batch row count: {batch_df.count()}")

def plot_feature_histograms(df, features, bins=30, save_path="特征分布直方图.png"):
    """
    Plot histograms of specified columns in Spark DataFrame (non-sampled version, optimized for speed)
    :param df: Spark DataFrame
    :param features: List of column names to plot
    :param bins: Number of histogram bins
    :param save_path: Image save path
    """

    
    num_features = len(features)
    if num_features == 0:
        print("No features specified for plotting")
        return
    
    pandas_df = df.select(features).dropna().toPandas()
    
    cols = 3
    rows = (num_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        values = pandas_df[feature].values
        
        axes[i].hist(values, bins=bins, color="skyblue", edgecolor="black")
        axes[i].set_title(f"{feature} Distribution", fontsize=12)
        axes[i].set_xlabel(feature, fontsize=10)
        axes[i].set_ylabel("Frequency", fontsize=10)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", format="png")
    print(f"Feature distribution histograms saved to: {save_path}")
    
def plot_correlation_heatmap(df, features, target, save_name="特征与等待时间相关性热力图.png"):
    """
    Plot correlation heatmap between features and target variable
    :param df: Spark DataFrame
    :param features: List of feature columns
    :param target: Target variable column name
    :param save_name: Save name for the image
    """
    columns = features + [target]
    
    pandas_df = df.select(columns).sample(withReplacement=False, fraction=0.1, seed=42).toPandas()
    
    corr_matrix = pandas_df.corr()
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title("Correlation Heatmap: Features vs Wait Time", fontsize=16)
    
    plt.tight_layout()
    
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    print(f"Correlation heatmap saved as: {save_name}")

def plot_hourly_trend(df_spark, save_path="hourly_wait_trend.png"):
    """
    Plot hourly wait time trend for weekdays vs weekends
    :param df_spark: Spark DataFrame (must contain request_hour, is_weekend, wait_minutes columns)
    :param save_path: Image save path
    """
    df_pandas = df_spark.select("request_hour", "is_weekend", "wait_minutes").toPandas()
    
    hourly_stats = df_pandas.groupby(["request_hour", "is_weekend"])["wait_minutes"].mean().reset_index()
    
    weekday_data = hourly_stats[hourly_stats["is_weekend"] == 0]
    weekend_data = hourly_stats[hourly_stats["is_weekend"] == 1]
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=weekday_data, 
        x="request_hour", 
        y="wait_minutes", 
        label="Weekday (is_weekend=0)", 
        color="steelblue", 
        marker="o"
    )
    sns.lineplot(
        data=weekend_data, 
        x="request_hour", 
        y="wait_minutes", 
        label="Weekend (is_weekend=1)", 
        color="orange", 
        marker="o"
    )
    
    plt.title("Hourly Wait Time Trend: Weekday vs Weekend", fontsize=14)
    plt.xlabel("Hour of Day (0-23)", fontsize=12)
    plt.ylabel("Average Wait Minutes", fontsize=12)
    plt.xticks(range(0, 24))
    plt.legend(title="Day Type", fontsize=10, title_fontsize=12)
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Hourly trend chart saved to: {save_path}")

def plot_top10_avg_wait_by_zone(df_spark, save_path="各地区平均等待时间Top10.png"):
    """
    Plot top 10 bar chart of average wait time by zone (PULocationZone)
    with zone name formatted as ({PUBorough}) PULocationZone
    :param df_spark: Spark DataFrame containing PULocationZone, PUBorough and wait_minutes fields
    :param save_path: Image save path
    """
    avg_wait_by_zone = df_spark.groupBy("PULocationZone", "PUBorough") \
        .agg(F.avg("wait_minutes").alias("avg_wait_minutes")) \
        .orderBy(F.desc("avg_wait_minutes")) \
        .limit(10)

    avg_wait_by_zone_pd = avg_wait_by_zone.toPandas()

    avg_wait_by_zone_pd["formatted_zone"] = avg_wait_by_zone_pd.apply(
        lambda row: f"({row['PUBorough']}) {row['PULocationZone']}",
        axis=1
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        x="avg_wait_minutes",
        y="formatted_zone",
        data=avg_wait_by_zone_pd,
        palette="viridis"
    )
    plt.title("Top 10 Zones by Average Wait Minutes", fontsize=14)
    plt.xlabel("Average Wait Minutes", fontsize=12)
    plt.ylabel("Pickup Zone (PULocationZone)", fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    print(f"Top 10 zones by average wait time chart saved to: {save_path}")

def plot_monthly_wait_trend(df_spark, save_path="月度等待时间趋势.png"):
    monthly_stats = df_spark.groupBy("trip_month") \
        .agg(F.avg("wait_minutes").alias("avg_wait")) \
        .orderBy("trip_month") \
        .toPandas()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="trip_month", y="avg_wait", data=monthly_stats, marker="o", color="darkorange")
    plt.xticks(range(1, 7))
    plt.title("Monthly Average Wait Time Trend", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Wait Minutes", fontsize=12)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300)

def plot_wait_time_geography(avg_wait_by_zone, shapefile_path, save_path="平均等待时间地理分布图.png"):
    """
    Plot geographical distribution heatmap of average wait time by zone
    
    Parameters:
        avg_wait_by_zone: Spark DataFrame containing "PULocationZone" and "avg_wait_minutes" columns
        shapefile_path: str, path to taxi zone shapefile
        save_path: str, image save path
    """
    wait_time_df = avg_wait_by_zone.toPandas()
    
    zones = gpd.read_file(shapefile_path)
    
    zones = zones.merge(
        wait_time_df,
        left_on="zone",
        right_on="PULocationZone",
        how="left"
    )
    
    zones["avg_wait_minutes"] = zones["avg_wait_minutes"].fillna(0)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    zones.plot(
        column="avg_wait_minutes",
        cmap="OrRd",
        linewidth=0.8,
        edgecolor="gray",
        legend=True,
        ax=ax
    )
    
    plt.title("Average Wait Time by NYC Taxi Zone (minutes)", fontsize=16)
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Geographical distribution map saved to: {save_path}")

if __name__ == "__main__":
    spark = init_spark_session()
    

    df_final_encoded, weather_mapping = load_and_preprocess_data(spark, "./data",sample_ratio=1.0)
    print(weather_mapping)
    output_path = "G:/processed_data.parquet"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    df_final_encoded.write.parquet(output_path, mode="overwrite")

    df_final_encoded.show(2)
    print("\n", df_final_encoded.columns)
    df_final_encoded = df_final_encoded.sample(withReplacement=False, fraction=0.1, seed=42)
    
    features = ["weather_code", "temp_num", "wind_num", "humidity_num", "barometer_num", "visibility_num","PULocationID","request_hour","is_weekend"]
    
    avg_wait_by_zone = df_final_encoded.groupBy("PULocationZone", "PUBorough") \
            .agg(F.avg("wait_minutes").alias("avg_wait_minutes")) \
            .orderBy(F.desc("avg_wait_minutes")).select("PULocationZone", "avg_wait_minutes")
    
    plot_feature_histograms(df_final_encoded, ["weather_code", "temp_num", "wind_num", "humidity_num", "barometer_num", "visibility_num"])
    plot_correlation_heatmap(df_final_encoded, features, 'wait_minutes')
    plot_hourly_trend(df_spark=df_final_encoded, save_path="工作日与非工作日小时级等待时间趋势图.png" )
    plot_top10_avg_wait_by_zone( df_spark=df_final_encoded,  save_path="各地区平均等待时间Top10.png"  )
    plot_monthly_wait_trend(df_final_encoded)
    plot_wait_time_geography(avg_wait_by_zone=avg_wait_by_zone,shapefile_path=r"F:\120\taxi\taxi_zones\taxi_zones.shp", save_path="NYC出租车区域平均等待时间分布图.png")