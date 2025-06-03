#!/usr/bin/env python3
"""
Service Registry for Arc Detection Orchestrator

Manages service discovery, health monitoring, and inter-service communication
for the distributed arc detection system.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

@dataclass
class ServiceInfo:
    """Information about a registered service"""
    name: str
    host: str
    port: int
    health_endpoint: str
    status: str = "unknown"  # unknown, healthy, unhealthy, stopped
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class ServiceRegistry:
    """Central service registry for orchestrating microservices"""
    
    def __init__(self, health_check_interval: int = 30):
        self.services: Dict[str, ServiceInfo] = {}
        self.health_check_interval = health_check_interval
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.health_monitor_thread = None
        
    def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a new service"""
        try:
            service_info.last_heartbeat = datetime.now()
            self.services[service_info.name] = service_info
            self.logger.info(f"Registered service: {service_info.name} at {service_info.host}:{service_info.port}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register service {service_info.name}: {e}")
            return False
    
    def deregister_service(self, service_name: str) -> bool:
        """Remove a service from the registry"""
        try:
            if service_name in self.services:
                del self.services[service_name]
                self.logger.info(f"Deregistered service: {service_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to deregister service {service_name}: {e}")
            return False
    
    def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get information about a specific service"""
        return self.services.get(service_name)
    
    def get_healthy_services(self) -> List[ServiceInfo]:
        """Get list of all healthy services"""
        return [service for service in self.services.values() if service.status == "healthy"]
    
    def get_all_services(self) -> Dict[str, ServiceInfo]:
        """Get all registered services"""
        return self.services.copy()
    
    def update_heartbeat(self, service_name: str) -> bool:
        """Update the heartbeat timestamp for a service"""
        try:
            if service_name in self.services:
                self.services[service_name].last_heartbeat = datetime.now()
                if self.services[service_name].status == "unhealthy":
                    self.services[service_name].status = "healthy"
                    self.logger.info(f"Service {service_name} recovered")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to update heartbeat for {service_name}: {e}")
            return False
    
    def start_health_monitoring(self):
        """Start the background health monitoring thread"""
        if not self.running:
            self.running = True
            self.health_monitor_thread = threading.Thread(target=self._health_monitor_loop)
            self.health_monitor_thread.daemon = True
            self.health_monitor_thread.start()
            self.logger.info("Started health monitoring")
    
    def stop_health_monitoring(self):
        """Stop the background health monitoring"""
        self.running = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join()
        self.logger.info("Stopped health monitoring")
    
    def _health_monitor_loop(self):
        """Background loop for monitoring service health"""
        while self.running:
            try:
                self._check_all_services_health()
                time.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
    
    def _check_all_services_health(self):
        """Check health of all registered services"""
        current_time = datetime.now()
        timeout_threshold = timedelta(seconds=self.health_check_interval * 2)
        
        for service_name, service_info in self.services.items():
            if service_info.last_heartbeat:
                time_since_heartbeat = current_time - service_info.last_heartbeat
                if time_since_heartbeat > timeout_threshold:
                    if service_info.status != "unhealthy":
                        service_info.status = "unhealthy"
                        self.logger.warning(f"Service {service_name} marked as unhealthy")
                elif service_info.status == "unknown":
                    service_info.status = "healthy"
    
    def export_registry_state(self) -> Dict[str, Any]:
        """Export current registry state for persistence"""
        return {
            "timestamp": datetime.now().isoformat(),
            "services": {
                name: {
                    **asdict(service),
                    "last_heartbeat": service.last_heartbeat.isoformat() if service.last_heartbeat else None
                }
                for name, service in self.services.items()
            }
        }
    
    def import_registry_state(self, state: Dict[str, Any]) -> bool:
        """Import registry state from persistence"""
        try:
            services_data = state.get("services", {})
            for name, service_data in services_data.items():
                service_data = service_data.copy()
                if service_data.get("last_heartbeat"):
                    service_data["last_heartbeat"] = datetime.fromisoformat(service_data["last_heartbeat"])
                
                service_info = ServiceInfo(**service_data)
                self.services[name] = service_info
            
            self.logger.info(f"Imported {len(services_data)} services from state")
            return True
        except Exception as e:
            self.logger.error(f"Failed to import registry state: {e}")
            return False


# Example service configurations
KNOWN_SERVICES = {
    "data_processor": {
        "port": 5000,
        "health_endpoint": "/health",
        "description": "Main data processing service"
    },
    "transient_viewer": {
        "port": 5001, 
        "health_endpoint": "/health",
        "description": "Transient visualization service"
    },
    "segment_visualizer": {
        "port": 5002,
        "health_endpoint": "/health", 
        "description": "Data segment visualization service"
    },
    "cache_manager": {
        "port": 5003,
        "health_endpoint": "/health",
        "description": "Caching and data management service"
    }
}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    registry = ServiceRegistry()
    
    # Register some example services
    for service_name, config in KNOWN_SERVICES.items():
        service = ServiceInfo(
            name=service_name,
            host="localhost",
            port=config["port"],
            health_endpoint=config["health_endpoint"],
            metadata={"description": config["description"]}
        )
        registry.register_service(service)
    
    # Start health monitoring
    registry.start_health_monitoring()
    
    try:
        # Keep running for demonstration
        time.sleep(60)
        
        # Print current status
        print("Current registry state:")
        for name, service in registry.get_all_services().items():
            print(f"  {name}: {service.status} at {service.host}:{service.port}")
            
    finally:
        registry.stop_health_monitoring()