from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import argparse

def compute_map_features(agent_df, map_df):
    map_exploded = map_df.withColumn("point", F.explode(F.arrays_zip("polyline_x", "polyline_y"))) \
                         .select("scene_id", "feature_id", F.col("point.polyline_x").alias("mx"), F.col("point.polyline_y").alias("my"))

    current_agents = agent_df.select(
        "scene_id", "agent_id", 
        F.col("x")[10].alias("ax"), 
        F.col("y")[10].alias("ay")
    )
    
    joined = current_agents.join(map_exploded, "scene_id")
    
    joined = joined.withColumn("dist_sq", F.pow(F.col("ax") - F.col("mx"), 2) + F.pow(F.col("ay") - F.col("my"), 2))
    
    stats = joined.groupBy("scene_id", "agent_id").agg(F.min("dist_sq").alias("min_dist_sq"))
    stats = stats.withColumn("map_dist", F.sqrt(F.col("min_dist_sq"))).drop("min_dist_sq")
    
    result = agent_df.join(stats, ["scene_id", "agent_id"], "left")
    result = result.fillna(999.0, subset=["map_dist"])
    
    return result

def main():
    spark = SparkSession.builder.appName("MapFeatureExtraction").getOrCreate()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_batch", type=int, default=0)
    parser.add_argument("--end_batch", type=int, default=55)
    args = parser.parse_args()

    BUCKET = "waymo-motion-pipeline-ashonfire"
    
    for i in range(args.start_batch, args.end_batch):
        batch_id = f"batch_{i:02d}"
        print(f"Processing {batch_id}...")
        
        input_path = f"gs://{BUCKET}/processed_full/{batch_id}"
        output_path = f"gs://{BUCKET}/processed_enriched/{batch_id}/agents"
        
        try:
            agents = spark.read.parquet(f"{input_path}/agents")
            maps = spark.read.parquet(f"{input_path}/maps")
            
            enriched = compute_map_features(agents, maps)
            
            enriched.write.mode("overwrite").parquet(output_path)
            print(f"Written enriched agents to {output_path}")
            
        except Exception as e:
            print(f"Skipping {batch_id}: {e}")

if __name__ == "__main__":
    main()
