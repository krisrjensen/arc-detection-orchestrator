#!/bin/bash
# Shutdown All Arc Detection Servers - 20250601_143900_0_0_1_9
# Cleanly stops all running arc detection services

echo "=== SHUTTING DOWN ARC DETECTION SYSTEM ==="
echo

# Function to check if a port is in use
check_port() {
    local port=$1
    lsof -ti:$port >/dev/null 2>&1
}

# Function to gracefully shutdown a service on a specific port
shutdown_service() {
    local port=$1
    local name=$2
    
    echo -n "Shutting down $name (port $port)... "
    
    if check_port $port; then
        # Get PIDs using the port
        local pids=$(lsof -ti:$port)
        
        if [ -n "$pids" ]; then
            # Send SIGTERM first for graceful shutdown
            echo $pids | xargs kill -TERM 2>/dev/null
            
            # Wait up to 5 seconds for graceful shutdown
            local count=0
            while [ $count -lt 5 ] && check_port $port; do
                sleep 1
                ((count++))
            done
            
            # If still running, force kill
            if check_port $port; then
                echo $pids | xargs kill -KILL 2>/dev/null
                sleep 1
            fi
            
            if check_port $port; then
                echo "❌ Failed to stop"
            else
                echo "✅ Stopped"
            fi
        else
            echo "✅ Not running"
        fi
    else
        echo "✅ Not running"
    fi
}

# Shutdown all services
shutdown_service 5020 "Database Browser Tool"
shutdown_service 5025 "Cache Configuration Server"
shutdown_service 5030 "Enhanced Data Cleaning Tool"
shutdown_service 5031 "Transient Prediction Viewer"
shutdown_service 5032 "Enhanced Data Segment Visualizer"
shutdown_service 5034 "Segment Verification Tool"

echo
echo "=== CLEANUP ==="

# Kill any remaining python processes that might be hanging
echo -n "Cleaning up any remaining python server processes... "
pkill -f "python.*port.*50[2-3][0-9]" 2>/dev/null
pkill -f "python.*--port.*50[2-3][0-9]" 2>/dev/null
pkill -f "python.*enhanced_data_cleaning_tool" 2>/dev/null
pkill -f "python.*cache_config_server" 2>/dev/null
pkill -f "python.*transient_prediction_viewer" 2>/dev/null
pkill -f "python.*data_segment_visualizer" 2>/dev/null
pkill -f "python.*segment_verification_tool" 2>/dev/null
pkill -f "python.*browse_database" 2>/dev/null
echo "✅ Done"

# Remove any leftover PID files
echo -n "Removing PID files... "
rm -f *.pid 2>/dev/null
echo "✅ Done"

echo
echo "=== FINAL STATUS CHECK ==="
active_services=0
for port in 5020 5025 5030 5031 5032 5034; do
    if check_port $port; then
        echo "⚠️  Port $port still in use"
        ((active_services++))
    else
        echo "✅ Port $port available"
    fi
done

echo
if [ $active_services -eq 0 ]; then
    echo "🎉 All arc detection services successfully shutdown"
    echo "🔄 Ready for restart or maintenance"
else
    echo "⚠️  $active_services services still running"
    echo "💡 You may need to manually kill remaining processes"
fi

echo
echo "=== SHUTDOWN COMPLETE ==="