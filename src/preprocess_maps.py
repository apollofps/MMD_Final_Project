from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
import argparse

def compute_map_features(agent_df, map_df):
    """
    Computes distance to nearest lane center for each agent.
    """
    # 1. Flatten Map Polylines to (scene_id, feature_id, x, y)
    # Using arrays_zip (Spark 2.4+ usually supports it, or we use posexplode)
    # If arrays_zip is not available, we can rely on python UDF or simply explode one array?
    # Let's assume standard PySpark environment (2.4/3.0)
    
    # Simple Flatten: Explode X, then Explode Y? Risk of mismatch if not ordered.
    # Safe way: Create Struct Array if possible.
    # Or just use Python UDF to flatten?
    
    # Let's use a simpler approach: Take just the first point of each lane for approximate "Lane Proximity".
    # This avoids the massive explosion (N points per polyline).
    # "Distance to nearest lane START" is a decent proxy for "On road vs Off road".
    
    # Better: Explode but limiting to 5 points per lane?
    # Let's simple explode.
    
    # Note: On Dataproc 2.0 (Spark 3.1), arrays_zip is available.
    map_exploded = map_df.withColumn("point", F.explode(F.arrays_zip("polyline_x", "polyline_y"))) \
                         .select("scene_id", "feature_id", F.col("point.polyline_x").alias("mx"), F.col("point.polyline_y").alias("my"))

    # Optimization: Filter map points? No, we need all potential lanes.
    
    # 2. Key Data: Agents at t=10 (Current Position)
    current_agents = agent_df.select(
        "scene_id", "agent_id", 
        F.col("x")[10].alias("ax"), 
        F.col("y")[10].alias("ay")
    )
    
    # 3. Join (Expensive!)
    # Optimization: Broadcast current_agents if small? No, agents are large too.
    joined = current_agents.join(map_exploded, "scene_id")
    
    # 4. Compute Squared Dist
    joined = joined.withColumn("dist_sq", F.pow(F.col("ax") - F.col("mx"), 2) + F.pow(F.col("ay") - F.col("my"), 2))
    
    # 5. Min Dist per Agent
    stats = joined.groupBy("scene_id", "agent_id").agg(F.min("dist_sq").alias("min_dist_sq"))
    stats = stats.withColumn("map_dist", F.sqrt(F.col("min_dist_sq"))).drop("min_dist_sq")
    
    # 6. Join back to original
    result = agent_df.join(stats, ["scene_id", "agent_id"], "left")
    result = result.fillna(999.0, subset=["map_dist"])
    
    return result

def main():
    spark = SparkSession.builder.appName("MapFeatureExtraction").getOrCreate()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_batch", type=int, default=0)
    parser.add_argument("--end_batch", type=int, default=55)
    args = parser.parse_args()

    # Process Batches based on args
    BUCKET = "waymo-motion-pipeline-ashonfire"
    
    for i in range(args.start_batch, args.end_batch):
        batch_id = f"batch_{i:02d}"
        print(f"Processing {batch_id}...")
        
        input_path = f"gs://{BUCKET}/processed_full/{batch_id}"
        output_path = f"gs://{BUCKET}/processed_enriched/{batch_id}/agents"
        
        # Read
        try:
            agents = spark.read.parquet(f"{input_path}/agents")
            maps = spark.read.parquet(f"{input_path}/maps")
            
            # Compute
            enriched = compute_map_features(agents, maps)
            
            # Write
            enriched.write.mode("overwrite").parquet(output_path)
            print(f"Written enriched agents to {output_path}")
            
        except Exception as e:
            print(f"Skipping {batch_id}: {e}")

if __name__ == "__main__":
    main()
