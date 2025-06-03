#!/bin/bash

# Simple Arc Detection System Shutdown

echo "=== SHUTTING DOWN SIMPLE ARC DETECTION SYSTEM ==="

# Function to stop a service by PID file
stop_service() {
    local service_name=$1
    local pid_file="${service_name,,}_simple.pid"
    local port=$2
    
    echo "Stopping $service_name (port $port)..."
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            sleep 1
            if kill -0 $pid 2>/dev/null; then
                kill -9 $pid 2>/dev/null
            fi
            echo "✅ $service_name stopped"
        else
            echo "⚠️  $service_name was not running"
        fi
        rm -f "$pid_file"
    else
        # Try to kill by port
        local pids=$(lsof -ti :$port)
        if [ -n "$pids" ]; then
            kill $pids 2>/dev/null
            echo "✅ $service_name stopped (by port)"
        else
            echo "✅ $service_name not running"
        fi
    fi
}

# Stop all services
stop_service "Enhanced Data Cleaning Tool" 5030
stop_service "Simple Transient Viewer" 5031
stop_service "Simple Segment Visualizer" 5032
stop_service "Segment Verification Tool" 5034

# Clean up any remaining Python processes
echo ""
echo "=== CLEANUP ==="
echo "Cleaning up any remaining python server processes..."
pkill -f "enhanced_data_cleaning_tool.py" 2>/dev/null
pkill -f "simple_transient_viewer" 2>/dev/null
pkill -f "simple_data_segment_visualizer" 2>/dev/null
pkill -f "segment_verification_tool" 2>/dev/null
echo "✅ Done"

# Remove PID files
echo "Removing PID files..."
rm -f *_simple.pid
echo "✅ Done"

# Final status check
echo ""
echo "=== FINAL STATUS CHECK ==="
ports=(5030 5031 5032 5034)
for port in "${ports[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  Port $port still in use"
    else
        echo "✅ Port $port available"
    fi
done

echo ""
echo "🎉 Simple arc detection system shutdown complete"
echo "🔄 Ready for restart with ./start_simple_system_20250601_164800_0_0_1_17.sh"