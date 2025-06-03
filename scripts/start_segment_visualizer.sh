#!/bin/bash
"""
Start Segment Visualizer - Automatically find available port
"""

echo "=== STARTING SEGMENT VISUALIZER ==="
echo "Trying to start on preferred ports..."

# Try ports in order of preference
for port in 5032 5034 5035 5036 5037; do
    echo "Trying port $port..."
    
    # Check if port is available
    if ! nc -z localhost $port 2>/dev/null; then
        echo "Port $port is available, starting server..."
        python3 segment_visualization_server_20250530_023000_0_0_1_1.py --port $port
        exit 0
    else
        echo "Port $port is in use, trying next..."
    fi
done

echo "❌ No available ports found in range 5032-5037"
echo "Please stop other services or restart your system"
exit 1