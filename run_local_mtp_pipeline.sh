#!/bin/bash

# Full Local Pipeline (MTP)
# 1. Train (10 Epochs)
# 2. Evaluate (Batch 50)

echo "🚀 Starting Full MTP Pipeline on M4 Mac..."

# Kill any existing training
pkill -f "train_local_mtp.py"
sleep 2

echo "🧠 Step 1: Training MTP Model..."
# Run unbuffered, log to file
python3 -u train_local_mtp.py > train_mtp_full.log 2>&1

echo "✅ Training Complete."
echo "🔮 Step 2: Evaluating MTP Model..."

python3 eval_local_mtp.py > eval_mtp_full.log 2>&1

echo "🎉 Pipeline Finished."
cat eval_mtp_full.log
