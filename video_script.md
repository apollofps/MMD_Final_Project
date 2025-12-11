# Video Script: Mining Massive Datasets Final Project
**Title**: Scaling Autonomous Trajectory Prediction  
**Duration**: 5 Minutes  
**Format**: Two Narrators (Split by Half)

---

## PART 1: Data Engineering & Model (Narrator A - 0:00 to 2:30)

### Introduction (0:00 - 0:30)
> "Hey everyone, welcome to our Mining Massive Datasets final project. I'm [Name A], and I'll walk you through how we tackled motion prediction at scale."
>
> "Our challenge: build a trajectory prediction system using the 2-terabyte Waymo Open Motion Dataset on a minimal cloud budget. Let me show you how we did it."

**[SCREEN: Show Waymo Dataset website or GCS bucket listing]**

---

### The Scale Problem (0:30 - 1:00)
> "The Waymo dataset has over 200,000 driving scenes with full trajectory and map data. Loading this into memory crashes any normal machine instantly."
>
> "We needed a distributed approach, so we deployed Apache Spark on Google Cloud Dataproc."

**[SCREEN: Show terminal with file listing or Dataproc console]**

---

### Spark Preprocessing (1:00 - 1:45)
> "Here's our preprocessing script. We use micro-batching—splitting the 1000 raw files into 100 independent batches. Each batch extracts agent trajectories and computes the distance to the nearest lane center."

**[SCREEN: Open `preprocess_maps.py` in VS Code, scroll through `compute_map_features` function]**

> "This map feature is critical. It tells the model where the road actually is."

---

### The MTP Model (1:45 - 2:30)
> "For the AI side, we built a Multi-Trajectory Prediction network. The key insight is that at intersections, cars can go multiple directions. A standard model predicts the average—often straight into a wall."
>
> "Our MTP model outputs three distinct paths with probabilities. We train with Winner-Takes-All loss, only penalizing the closest prediction. This forces each head to specialize in different maneuvers."

**[SCREEN: Open `src/model_mtp.py`, highlight the class definition and `mtp_loss` function]**

> "Now I'll hand it over to [Name B] to walk through our system architecture and results."

---

## PART 2: Architecture, Results & Conclusion (Narrator B - 2:30 to 5:00)

### System Architecture (2:30 - 3:15)
> "Thanks [Name A]. I'm [Name B]. Let me show you the full picture of how our system works."
>
> "Here's our pipeline diagram. We have a hybrid architecture that separates data processing from model training."

**[SCREEN: Show the Mermaid architecture diagram from Final_Report.md]**

> "On the left, raw Waymo data flows into Google Cloud Dataproc where Spark handles the heavy ETL work—deserializing Protobufs, extracting map features, and writing optimized Parquet files."
>
> "Those files get downloaded to our local machine, where we train the MTP model using Apple's Metal Performance Shaders. This hybrid approach lets us leverage cloud compute for data processing while keeping training costs at zero."

---

### Results (3:15 - 4:15)
> "Now for the results. We compared three approaches: a physics baseline using constant velocity, a single-mode LSTM, and our Multi-Trajectory Prediction model."

**[SCREEN: Show `model_comparison.png` bar chart]**

> "The baseline had a Final Displacement Error of over 11 meters—meaning after 8 seconds, predictions were 11 meters off target. The single-mode LSTM improved that to about 10 meters."
>
> "But our MTP model achieved just 2.93 meters. That's a 70% reduction in error."

**[SCREEN: Show `scaling_efficiency.png` plot]**

> "We also ran scaling experiments. Training on 50 batches versus 100 batches showed minimal improvement—the model had saturated its learning capacity. And interestingly, making the model bigger actually hurt performance due to overfitting."

---

### Conclusion (4:15 - 5:00)
> "So what did we learn? Three key takeaways."
>
> "First, you don't need a supercomputer. With smart distributed ETL and edge training, we processed 2 terabytes on a student budget."
>
> "Second, architecture matters more than scale. The switch to multi-modal prediction gave us 70% error reduction—far more than simply adding data or parameters."
>
> "Third, hybrid pipelines are the future. Separating data processing from model training lets you optimize each component independently."
>
> "Thanks for watching. Our code is open source on GitHub."

**[SCREEN: Show GitHub repo page with README visible]**

---

## Recording Checklist
- [ ] Waymo website or GCS console
- [ ] VS Code with `preprocess_maps.py` open
- [ ] VS Code with `src/model_mtp.py` open
- [ ] Architecture diagram from Final_Report.md
- [ ] Bar chart (`model_comparison.png`)
- [ ] Scaling plot (`scaling_efficiency.png`)
- [ ] GitHub repo page
