#!/bin/bash

# Simple Arc Detection System Starter - 20250601_164800_0_0_1_17
# NO CACHING - Direct generation of plots on demand
# Simple, reliable, no hanging issues

echo "=== STARTING SIMPLE ARC DETECTION SYSTEM ==="
echo "No caching - direct plot generation on demand"
echo ""

# Function to check if a port is in use
check_port() {
    local port=$1
    local service_name=$2
    
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  Port $port ($service_name) is already in use"
        return 1
    else
        echo "✅ Port $port ($service_name) is available"
        return 0
    fi
}

# Function to start a service
start_service() {
    local script=$1
    local port=$2
    local service_name=$3
    local log_file=$4
    
    echo "Starting $service_name on port $port..."
    
    if check_port $port "$service_name"; then
        # Start the service in background
        python3 "$script" --port $port > "$log_file" 2>&1 &
        local pid=$!
        
        # Wait a moment for startup
        sleep 2
        
        # Check if process is still running
        if kill -0 $pid 2>/dev/null; then
            echo "✅ $service_name started successfully (PID: $pid)"
            echo $pid > "${service_name,,}_simple.pid"
            return 0
        else
            echo "❌ $service_name failed to start"
            return 1
        fi
    else
        echo "⚠️  Port $port is occupied, skipping $service_name"
        return 1
    fi
}

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Start core services
echo "=== STARTING CORE SERVICES ==="

# 1. Enhanced Data Cleaning Tool (main interface)
start_service "enhanced_data_cleaning_tool.py" 5030 "Enhanced Data Cleaning Tool" "enhanced_data_cleaning_simple.log"

# 2. Simple Transient Viewer (no caching)
start_service "simple_transient_viewer_20250601_164700_0_0_1_16.py" 5031 "Simple Transient Viewer" "simple_transient_viewer.log"

# 3. Simple Data Segment Visualizer (no caching)
start_service "simple_data_segment_visualizer_20250601_164500_0_0_1_15.py" 5032 "Simple Segment Visualizer" "simple_segment_visualizer.log"

# 4. Segment Verification Tool
start_service "segment_verification_tool_20250531_153000_0_0_1_1.py" 5034 "Segment Verification Tool" "segment_verification_simple.log"

echo ""
echo "=== STARTUP COMPLETE ==="

# Final status check
echo "=== FINAL STATUS CHECK ==="
services=(
    "5030:Enhanced Data Cleaning Tool"
    "5031:Simple Transient Viewer" 
    "5032:Simple Segment Visualizer"
    "5034:Segment Verification Tool"
)

all_running=true

for service in "${services[@]}"; do
    port="${service%%:*}"
    name="${service##*:}"
    
    if lsof -i :$port > /dev/null 2>&1; then
        echo "✅ $name (port $port) - Running"
    else
        echo "❌ $name (port $port) - Not running"
        all_running=false
    fi
done

echo ""
if $all_running; then
    echo "🎉 All services started successfully!"
    echo ""
    echo "🌐 Access URLs:"
    echo "   Main Interface:     http://localhost:5030"
    echo "   Transient Viewer:   http://localhost:5031"
    echo "   Segment Visualizer: http://localhost:5032"
    echo "   Segment Verification: http://localhost:5034"
    echo ""
    echo "📱 Network Access (other devices):"
    local_ip=$(ifconfig en0 | grep "inet " | awk '{print $2}' 2>/dev/null || echo "N/A")
    if [ "$local_ip" != "N/A" ]; then
        echo "   Main Interface:     http://$local_ip:5030"
        echo "   Transient Viewer:   http://$local_ip:5031"
        echo "   Segment Visualizer: http://$local_ip:5032"
        echo "   Segment Verification: http://$local_ip:5034"
    fi
    echo ""
    echo "🚀 SIMPLE SYSTEM READY - No caching, no hangs!"
    echo "   When you navigate in the main tool, other tools generate plots on demand"
    echo "   Clean, direct, reliable visualization"
else
    echo "⚠️  Some services failed to start. Check log files for details."
fi

echo ""
echo "💡 To shutdown: Run ./shutdown_simple_system.sh"
echo "📊 To monitor: tail -f *.log"