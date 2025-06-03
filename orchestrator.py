#!/usr/bin/env python3
"""
Arc Detection System Orchestrator

Main orchestration service that manages the lifecycle of all microservices
in the distributed arc detection system.
"""

import json
import logging
import argparse
import signal
import sys
import time
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from service_registry import ServiceRegistry, ServiceInfo, KNOWN_SERVICES
from web_interface import RegistryWebInterface


class SystemOrchestrator:
    """Main orchestrator for managing the arc detection system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = self._load_config(config_file)
        self.registry = ServiceRegistry(health_check_interval=30)
        self.web_interface = RegistryWebInterface(self.registry, port=8000)
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.processes: Dict[str, subprocess.Popen] = {}
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load orchestrator configuration"""
        default_config = {
            "services": KNOWN_SERVICES,
            "startup_order": ["cache_manager", "data_processor", "transient_viewer", "segment_visualizer"],
            "shutdown_timeout": 30,
            "health_check_interval": 30,
            "max_restart_attempts": 3
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.error(f"Failed to load config file {config_file}: {e}")
        
        return default_config
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.stop_all_services()
        sys.exit(0)
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service"""
        try:
            if service_name in self.processes:
                self.logger.warning(f"Service {service_name} is already running")
                return True
            
            service_config = self.config["services"].get(service_name)
            if not service_config:
                self.logger.error(f"Unknown service: {service_name}")
                return False
            
            # Look for service start script
            script_path = Path("scripts") / f"start_{service_name}.sh"
            if not script_path.exists():
                # Try generic startup approach
                script_path = Path("scripts") / "start_all_servers_20250601_144000_0_0_1_10.sh"
            
            if script_path.exists():
                # Start the service process
                process = subprocess.Popen(
                    ["bash", str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self.processes[service_name] = process
                
                # Register with service registry
                service_info = ServiceInfo(
                    name=service_name,
                    host="localhost",
                    port=service_config["port"],
                    health_endpoint=service_config["health_endpoint"],
                    metadata=service_config
                )
                self.registry.register_service(service_info)
                
                self.logger.info(f"Started service: {service_name}")
                return True
            else:
                self.logger.error(f"No startup script found for service: {service_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service"""
        try:
            # Remove from registry first
            self.registry.deregister_service(service_name)
            
            # Stop the process
            if service_name in self.processes:
                process = self.processes[service_name]
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=self.config["shutdown_timeout"])
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing service: {service_name}")
                    process.kill()
                    process.wait()
                
                del self.processes[service_name]
                self.logger.info(f"Stopped service: {service_name}")
                return True
            else:
                self.logger.warning(f"Service {service_name} was not running")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to stop service {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all services in the configured order"""
        startup_order = self.config.get("startup_order", list(self.config["services"].keys()))
        
        success = True
        for service_name in startup_order:
            if not self.start_service(service_name):
                success = False
                self.logger.error(f"Failed to start {service_name}, continuing with others...")
            else:
                # Wait a bit between service starts
                time.sleep(2)
        
        return success
    
    def stop_all_services(self) -> bool:
        """Stop all running services"""
        # Stop in reverse order
        startup_order = self.config.get("startup_order", list(self.config["services"].keys()))
        
        success = True
        for service_name in reversed(startup_order):
            if not self.stop_service(service_name):
                success = False
        
        return success
    
    def get_system_status(self) -> Dict:
        """Get current status of all services"""
        return {
            "timestamp": datetime.now().isoformat(),
            "orchestrator_status": "running" if self.running else "stopped",
            "services": self.registry.export_registry_state()["services"],
            "process_count": len(self.processes)
        }
    
    def run(self):
        """Main orchestrator run loop"""
        self.logger.info("Starting Arc Detection System Orchestrator")
        self.running = True
        
        # Start service registry health monitoring
        self.registry.start_health_monitoring()
        
        # Start web interface in a separate thread
        import threading
        web_thread = threading.Thread(target=self.web_interface.run, kwargs={'debug': False})
        web_thread.daemon = True
        web_thread.start()
        self.logger.info("Web interface started on http://localhost:8000")
        
        # Start all services
        if self.start_all_services():
            self.logger.info("All services started successfully")
        else:
            self.logger.warning("Some services failed to start")
        
        try:
            # Main monitoring loop
            while self.running:
                self._monitor_services()
                time.sleep(30)  # Check every 30 seconds
                
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.cleanup()
    
    def _monitor_services(self):
        """Monitor service health and restart if needed"""
        for service_name, process in list(self.processes.items()):
            if process.poll() is not None:  # Process has terminated
                self.logger.warning(f"Service {service_name} has stopped unexpectedly")
                del self.processes[service_name]
                self.registry.deregister_service(service_name)
                
                # Attempt restart
                self.logger.info(f"Attempting to restart {service_name}")
                self.start_service(service_name)
    
    def cleanup(self):
        """Clean up resources before shutdown"""
        self.logger.info("Cleaning up orchestrator resources")
        self.stop_all_services()
        self.registry.stop_health_monitoring()
        self.running = False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Arc Detection System Orchestrator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--action", choices=["start", "stop", "status", "restart"], 
                       default="start", help="Action to perform")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = SystemOrchestrator(args.config)
    
    if args.action == "start":
        orchestrator.run()
    elif args.action == "stop":
        orchestrator.stop_all_services()
    elif args.action == "status":
        status = orchestrator.get_system_status()
        print(json.dumps(status, indent=2))
    elif args.action == "restart":
        orchestrator.stop_all_services()
        time.sleep(5)
        orchestrator.start_all_services()


if __name__ == "__main__":
    main()