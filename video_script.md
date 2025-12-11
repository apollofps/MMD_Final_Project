# Video Script: Mining Massive Datasets Final Project
**Title**: Scaling Autonomous Trajectory Prediction on Constraints  
**Duration**: ~5 Minutes  
**Narrators**: 2 (Alex - Data Eng, Sam - AI Researcher)

---

## Segment 1: The Challenge (0:00 - 1:00)

| Time | Speaker | Audio Script | Visual / Screen Record |
| :--- | :--- | :--- | :--- |
| **0:00** | **Alex** | Imagine predicting the future movement of every car in a city. Now imagine doing it with 2 Terabytes of data, but you only have a laptop and a small cloud budget. That was our challenge for the Mining Massive Datasets project. | **Visual**: Show the "Waymo Open Motion Dataset" Website splash page. Fade to a quick montage of self-driving cars moving. |
| **0:20** | **Sam** | Exactly. We tackled the Waymo Open Motion Dataset. It contains over 200,000 scenes of complex traffic. The industry standard is to use massive GPU clusters. We asked: Can we build a research-grade pipeline using a Hybrid Cloud and Edge approach? | **Visual**: Scroll through the raw TFRecord file list in Google Cloud Storage console (showing 1000s of files). |
| **0:45** | **Alex** | Spoiler alert: We did. And we improved the baseline error by 70%. Let’s show you how. | **Visual**: Show the final "minADE 2.07m" result highlighted in the `Final_Report.md`. |

---

## Segment 2: The Data Architecture (1:00 - 2:15)

| Time | Speaker | Audio Script | Visual / Screen Record |
| :--- | :--- | :--- | :--- |
| **1:00** | **Alex** | It starts with the Data Pipeline. We couldn't just load 2TB into memory. We built a distributed ETL pipeline on Google Cloud Dataproc using Apache Spark. | **Visual**: Show `walkthrough.md` Architecture Diagram (Mermaid graph). Tooltip over "Cloud Dataproc". |
| **1:20** | **Alex** | I wrote a custom micro-batching script `preprocess_maps.py`. It shreds the complex Protobuf data, extracts the geometric lane features, and flattens everything into Parquet files. | **Visual**: Open `preprocess_maps.py` in VS Code. Highlight the `compute_map_features` function and the SQL-like dataframe transformations. |
| **1:45** | **Sam** | This was critical. By converting to Parquet and pruning unused LiDAR data, Alex reduced the storage footprint by 60%. This allowed us to stream the data efficiently to our training node. | **Visual**: Show the terminal downloading files: `gsutil -m cp ...`. Show the file sizes dropping from GBs to MBs. |

---

## Segment 3: The Model (2:15 - 3:30)

| Time | Speaker | Audio Script | Visual / Screen Record |
| :--- | :--- | :--- | :--- |
| **2:15** | **Sam** | Now for the brain. We started with a standard LSTM model. It failed miserably. It generated an Average Displacement Error of 4 meters. | **Visual**: Show the plot `model_comparison.png`. Point to the "Baseline" bar. |
| **2:30** | **Sam** | The problem is "The Average". When a car approaches a fork, it can go Left or Right. A standard model predicts the average: straight into the divider. | **Visual**: Use a drawing tool (iPad or Paint) to sketch a Y-junction and draw a line going straight into the wall (labeled "L2 Loss Failure"). |
| **2:50** | **Sam** | So we built a Multi-Trajectory Prediction network—MTP. It outputs THREE distinct paths and their probabilities. We trained this locally on an M4 Mac using Apple's Metal Performance Shaders. | **Visual**: Open `src/model_mtp.py`. Scroll to `class MotionMTP`. Highlight `self.num_modes = 3`. |
| **3:10** | **Sam** | We used a "Winner-Takes-All" loss. We only punish the prediction that was *closest* to the truth, allowing the other heads to specialize in other possibilities. | **Visual**: Highlight the `mtp_loss` function in the code. Show the `torch.min` operation. |

---

## Segment 4: Results & Demo (3:30 - 4:15)

| Time | Speaker | Audio Script | Visual / Screen Record |
| :--- | :--- | :--- | :--- |
| **3:30** | **Alex** | Let's see it running. Here is the training loop on the full 100-batch dataset. You can see the loss dropping as it streams data from the SSD. | **Visual**: Screen record the terminal running `train_local_mtp.py`. Speed it up (timelapse) to show the Loss numbers falling. |
| **3:50** | **Sam** | And here are the final evaluation metrics. We achieved a minimum Average Displacement Error of 2.07 meters. That’s a massive jump from the 4-meter baseline. | **Visual**: Show the plain text output of `eval_local_mtp.py` showing `minADE: 2.07m`. |
| **4:05** | **Alex** | We even tried scaling the model size up to 512 hidden units, but found the smaller, efficient model actually performed better on this dataset scale. | **Visual**: Show `scaling_efficiency.png` plot. Point to the saturation curve. |

---

## Segment 5: Conclusion (4:15 - 5:00)

| Time | Speaker | Audio Script | Visual / Screen Record |
| :--- | :--- | :--- | :--- |
| **4:15** | **Alex** | So, what did we learn? You don't need a supercomputer to do massive data mining. You need smart architecture. | **Visual**: Camera pans back to Host A. |
| **4:30** | **Sam** | By combining Spark for the heavy lifting and optimized Edge hardware for the intelligence, we built a pipeline that scales linearly. | **Visual**: Camera pans back to Host B. |
| **4:45** | **Both** | This is the future of efficient AI. Thanks for watching. | **Visual**: Fade to Black. Text: "Mining Massive Datasets Fall 2025". |
