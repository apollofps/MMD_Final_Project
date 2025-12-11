import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set style for research-grade plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
OUTPUT_DIR = "/Users/apollo/.gemini/antigravity/brain/49b981d9-7429-4264-bb7a-231cd4ea72c3"

def  plot_model_comparison():
    data = {
        "Model": ["Baseline\n(CV)", "Single-Mode\n(LSTM v3)", "Multi-Modal\n(MTP v5)"],
        "minADE": [4.07, 3.58, 2.07],
        "minFDE": [11.28, 9.89, 2.93]
    }
    df = pd.DataFrame(data)
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Error (meters)")

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x="Model", y="Error (meters)", hue="Metric", data=df_melt, palette="viridis")
    
    # Annotate improvements
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=10, fontweight='bold')

    plt.title("Motion Prediction Performance: Baseline vs. Deep Learning", fontsize=14, pad=20)
    plt.ylim(0, 13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=300)
    print("Generated model_comparison.png")

def plot_scaling_efficiency():
    # Training Time vs Data Size (Hybrid Pipeline)
    # 5 Batches ~ 15 mins
    # 50 Batches ~ 100 mins
    # 100 Batches ~ 210 mins
    
    batches = [5, 50, 100]
    time_min = [15, 100, 210]
    ade = [4.19, 2.07, 2.11]
    
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Dataset Size (Batches)')
    ax1.set_ylabel('Training Time (min)', color=color)
    ax1.plot(batches, time_min, marker='o', color=color, linewidth=3, label="Training Time")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Validation minADE (m)', color=color)
    ax2.plot(batches, ade, marker='s', linestyle='--', color=color, linewidth=3, label="Error (minADE)")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 5)

    plt.title("Scaling Analysis: Cost vs. Performance", fontsize=14, pad=20)
    fig.tight_layout() 
    plt.savefig(f"{OUTPUT_DIR}/scaling_efficiency.png", dpi=300)
    print("Generated scaling_efficiency.png")

if __name__ == "__main__":
    plot_model_comparison()
    plot_scaling_efficiency()
