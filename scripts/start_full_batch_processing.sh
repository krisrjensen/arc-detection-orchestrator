#!/bin/bash
# Start full batch transient detection processing
# This will run in the background and process all 1769 files

cd "$(dirname "$0")"

echo "=== STARTING FULL BATCH TRANSIENT DETECTION ==="
echo "Processing all 1769 files in V3 database"
echo "This will run in the background and may take several hours"
echo ""
echo "Log file: batch_transient_detection_log.txt"
echo "Monitor progress with: tail -f batch_transient_detection_log.txt"
echo ""

# Start the batch processor in background
nohup python3 batch_transient_detection_20250530_015500_0_0_1_1.py > batch_transient_output.log 2>&1 &

# Get the process ID
PID=$!
echo "Batch processing started with PID: $PID"
echo "To stop processing: kill $PID"
echo ""
echo "The process will continue running even if you close this terminal."
echo "Check batch_transient_detection_log.txt for detailed progress."

# Write PID to file for easy reference
echo $PID > batch_transient_detection.pid
echo "PID saved to batch_transient_detection.pid"