#!/bin/bash
# Start All Arc Detection Servers - 20250601_144000_0_0_1_10
# Comprehensive startup script for all arc detection services
# Always includes Database Browser Tool (port 5020)

echo "=== STARTING COMPLETE ARC DETECTION SYSTEM ==="
echo

# Setup directories
echo "Setting up directories..."
mkdir -p temp_plots temp_transient_plots temp_segment_plots temp_verification_plots cache
echo "✓ Directories ready"
echo

# Function to check if a port is available
check_port_available() {
    local port=$1
    ! lsof -ti:$port >/dev/null 2>&1
}

# Function to start a service and wait for it to be ready
start_service() {
    local script=$1
    local port=$2
    local name=$3
    local extra_args=$4
    
    echo -n "Starting $name on port $port... "
    
    if ! check_port_available $port; then
        echo "⚠️  Port $port already in use, stopping existing service"
        lsof -ti:$port | xargs kill -TERM 2>/dev/null
        sleep 2
        if ! check_port_available $port; then
            lsof -ti:$port | xargs kill -KILL 2>/dev/null
            sleep 1
        fi
    fi
    
    # Start the service
    if [ -n "$extra_args" ]; then
        python3 "$script" $extra_args > "${script%.*}.log" 2>&1 &
    else
        python3 "$script" --port $port > "${script%.*}.log" 2>&1 &
    fi
    
    local pid=$!
    echo $pid > "${script%.*}.pid"
    
    # Wait for service to start (up to 10 seconds)
    local count=0
    while [ $count -lt 10 ]; do
        if curl -s --connect-timeout 2 "http://localhost:$port" >/dev/null 2>&1; then
            echo "✅ Started successfully (PID: $pid)"
            return 0
        fi
        sleep 1
        ((count++))
    done
    
    echo "❌ Failed to start (check ${script%.*}.log)"
    return 1
}

echo "=== STARTING CORE SERVICES ==="

# Start Database Browser Tool (port 5020) - ALWAYS INCLUDED
start_service "browse_database.py" 5020 "Database Browser Tool" "--port 5020"

# Start Cache Configuration Server (port 5025)
start_service "cache_config_server_20250531_230100_0_0_1_1.py" 5025 "Cache Configuration Server"

# Start Enhanced Data Cleaning Tool (port 5030) - with parallel_motor_weak_arc support
start_service "enhanced_data_cleaning_tool.py" 5030 "Enhanced Data Cleaning Tool (with parallel_motor_weak_arc)"

# Start Transient Prediction Viewer (port 5031) - with right-click zoom
start_service "transient_prediction_viewer_20250526_125000_002_001_001_001.py" 5031 "Transient Prediction Viewer (Right-Click Zoom)"

# Start Enhanced Data Segment Visualizer (port 5032) - with caching
start_service "data_segment_visualizer_cached_20250531_230200_0_0_1_1.py" 5032 "Enhanced Data Segment Visualizer (Cached)"

# Start Segment Verification Tool (port 5034)
start_service "segment_verification_tool_20250531_153000_0_0_1_1.py" 5034 "Segment Verification Tool"

echo
echo "Waiting for services to initialize..."
sleep 3

echo
echo "=== SERVICE STATUS CHECK ==="
all_running=true
for port in 5020 5025 5030 5031 5032 5034; do
    if curl -s --connect-timeout 3 "http://localhost:$port" >/dev/null 2>&1; then
        case $port in
            5020) echo "✅ Database Browser Tool (port 5020): Running" ;;
            5025) echo "✅ Cache Configuration Server (port 5025): Running" ;;
            5030) echo "✅ Enhanced Data Cleaning Tool (port 5030): Running" ;;
            5031) echo "✅ Transient Prediction Viewer (port 5031): Running" ;;
            5032) echo "✅ Enhanced Data Segment Visualizer (port 5032): Running" ;;
            5034) echo "✅ Segment Verification Tool (port 5034): Running" ;;
        esac
    else
        case $port in
            5020) echo "❌ Database Browser Tool (port 5020): Failed" ;;
            5025) echo "❌ Cache Configuration Server (port 5025): Failed" ;;
            5030) echo "❌ Enhanced Data Cleaning Tool (port 5030): Failed" ;;
            5031) echo "❌ Transient Prediction Viewer (port 5031): Failed" ;;
            5032) echo "❌ Enhanced Data Segment Visualizer (port 5032): Failed" ;;
            5034) echo "❌ Segment Verification Tool (port 5034): Failed" ;;
        esac
        all_running=false
    fi
done

echo
if $all_running; then
    echo "🎉 === SYSTEM READY ==="
    echo
    echo "🗄️  Database Browser:           http://localhost:5020"
    echo "⚙️  Cache Configuration:        http://localhost:5025"
    echo "🧹 Enhanced Data Cleaning:     http://localhost:5030"
    echo "🎯 Transient Prediction:       http://localhost:5031"
    echo "📊 Data Segment Visualizer:    http://localhost:5032"
    echo "✅ Segment Verification:       http://localhost:5034"
    echo
    echo "📋 Service Architecture:"
    echo "   • Port 5020 provides database administration and browsing"
    echo "   • Port 5030 acts as the main coordination service"
    echo "   • Port 5031 syncs with 5030 for file selections (right-click zoom enabled)"
    echo "   • Port 5032 provides advanced segment management with caching"
    echo "   • Port 5034 verifies segment-transient alignment"
    echo "   • Port 5025 manages caching configuration (Nr/Nf parameters)"
    echo
    echo "🆕 New Features:"
    echo "   • Enhanced Data Cleaning Tool supports 'parallel_motor_weak_arc' (key '9')"
    echo "   • Segment configuration with enhanced sensitivity for weak arcs"
    echo "   • 75 segments with 5% overlap for better weak arc detection"
    echo
    echo "🚀 All services ready for high-speed data approval workflows!"
    echo
    echo "💡 To shutdown all services, run: ./shutdown_servers_20250601_143900_0_0_1_9.sh"
else
    echo "⚠️  Some services failed to start. Check log files:"
    echo "   • Database browser: browse_database.log"
    echo "   • Cache config: cache_config_server_20250531_230100_0_0_1_1.log"
    echo "   • Data cleaning: enhanced_data_cleaning_tool.log"
    echo "   • Transient viewer: transient_prediction_viewer_20250526_125000_002_001_001_001.log"
    echo "   • Segment visualizer: data_segment_visualizer_cached_20250531_230200_0_0_1_1.log"
    echo "   • Verification tool: segment_verification_tool_20250531_153000_0_0_1_1.log"
fi

echo
echo "=== STARTUP COMPLETE ==="

# Keep script running to monitor services (optional)
if $all_running; then
    echo
    echo "Press Ctrl+C to stop monitoring (services will continue running)"
    echo "Monitoring service status every 30 seconds..."
    echo
    
    trap 'echo ""; echo "Monitoring stopped. Services are still running."; echo "Use shutdown script to stop services."; exit 0' INT
    
    while true; do
        sleep 30
        failed_services=0
        for port in 5020 5025 5030 5031 5032 5034; do
            if ! curl -s --connect-timeout 2 "http://localhost:$port" >/dev/null 2>&1; then
                echo "⚠️  $(date '+%Y-%m-%d %H:%M:%S') - Service on port $port is not responding"
                ((failed_services++))
            fi
        done
        
        if [ $failed_services -eq 0 ]; then
            echo "✅ $(date '+%Y-%m-%d %H:%M:%S') - All services healthy"
        fi
    done
fi