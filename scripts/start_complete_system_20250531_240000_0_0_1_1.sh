#!/bin/bash

# Complete Arc Detection System Startup - Version 20250531_240000_0_0_1_1
# Launches all 6 components of the arc detection analysis system

echo "=== STARTING COMPLETE ARC DETECTION SYSTEM ==="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Create necessary directories
echo "Setting up directories..."
mkdir -p cache/segments
mkdir -p cache/plots  
mkdir -p cache/verification
mkdir -p temp_plots
mkdir -p temp_verification_plots
mkdir -p temp_transient_plots
echo "✓ Directories ready"

# Function to start a service in the background
start_service() {
    local script=$1
    local port=$2
    local name=$3
    local log_file=$4
    
    echo "Starting $name on port $port..."
    
    # Kill any existing process on this port
    lsof -ti:$port | xargs kill -9 2>/dev/null || true
    
    # Check if script exists
    if [ ! -f "$script" ]; then
        echo "✗ Script not found: $script"
        return 1
    fi
    
    # Start the service
    python3 "$script" > "$log_file" 2>&1 &
    local pid=$!
    
    # Wait a moment and check if it's still running
    sleep 3
    if kill -0 $pid 2>/dev/null; then
        echo "✓ $name started successfully (PID: $pid)"
        return 0
    else
        echo "✗ Failed to start $name"
        echo "Last 10 lines of log:"
        tail -10 "$log_file" 2>/dev/null || echo "No log available"
        return 1
    fi
}

# Start all services
echo ""
echo "=== STARTING CORE SERVICES ==="

# 1. Database Browser Tool (Port 5020)
start_service "database_browser_20250531_170900_0_0_1_1.py" 5020 "Database Browser Tool" "database_browser.log"

# 2. Cache Configuration Server (Port 5025)  
start_service "cache_config_server_20250531_230100_0_0_1_1.py" 5025 "Cache Configuration Server" "cache_config_server.log"

# 3. Enhanced Data Cleaning Tool (Port 5030)
start_service "enhanced_data_cleaning_tool.py" 5030 "Enhanced Data Cleaning Tool" "enhanced_data_cleaning.log"

# 4. Transient Prediction Viewer (Port 5031) - Right-Click Zoom Version
start_service "transient_prediction_viewer_20250526_125000_002_001_001_001.py" 5031 "Transient Prediction Viewer (Right-Click Zoom)" "transient_prediction_viewer_right_click.log"

# 5. Enhanced Data Segment Visualizer (Port 5032)  
start_service "data_segment_visualizer_cached_20250531_230200_0_0_1_1.py" 5032 "Enhanced Data Segment Visualizer" "data_segment_visualizer_cached.log"

# 6. Segment Verification Tool (Port 5034) - Use non-cached version to avoid hanging
start_service "segment_verification_tool_20250531_153000_0_0_1_1.py" 5034 "Segment Verification Tool" "segment_verification.log"

# Wait for services to fully initialize
echo ""
echo "Waiting for services to initialize..."
sleep 8

# Check service status
echo ""
echo "=== SERVICE STATUS ===="

check_service() {
    local port=$1
    local name=$2
    local test_path=${3:-"/"}
    
    if curl -s --max-time 5 "http://localhost:$port$test_path" > /dev/null 2>&1; then
        echo "✓ $name (port $port): Running"
        return 0
    else
        echo "✗ $name (port $port): Not responding"
        return 1
    fi
}

# Check all services
check_service 5020 "Database Browser Tool" "/"
check_service 5025 "Cache Configuration Server" "/"
check_service 5030 "Enhanced Data Cleaning Tool" "/status"
check_service 5031 "Transient Prediction Viewer" "/status"
check_service 5032 "Enhanced Data Segment Visualizer" "/status"
check_service 5034 "Segment Verification Tool" "/status"

echo ""
echo "=== SYSTEM READY ==="
echo ""
echo "🗄️  Database Browser:          http://localhost:5020"
echo "⚙️  Cache Configuration:       http://localhost:5025"
echo "🧹 Enhanced Data Cleaning:    http://localhost:5030"
echo "🎯 Transient Prediction:      http://localhost:5031"
echo "📊 Data Segment Visualizer:   http://localhost:5032"
echo "✅ Segment Verification:      http://localhost:5034"
echo ""
echo "📋 Service Architecture:"
echo "   • Port 5030 acts as the main coordination service"
echo "   • Port 5031 syncs with 5030 for file selections"
echo "   • Port 5032 provides advanced segment management with caching"
echo "   • Port 5034 verifies segment-transient alignment"
echo "   • Port 5025 manages caching configuration (Nr/Nf parameters)"
echo "   • Port 5020 provides database administration"
echo ""
echo "🚀 All services ready for high-speed data approval workflows!"
echo ""
echo "Press Ctrl+C to stop all services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping all services..."
    
    # Kill services by port
    for port in 5020 5025 5030 5031 5032 5034; do
        pid=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$pid" ]; then
            kill $pid 2>/dev/null
            echo "✓ Stopped service on port $port"
        fi
    done
    
    echo "All services stopped."
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
while true; do
    sleep 1
done