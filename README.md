# Arc Detection System Orchestrator

Service orchestration and management system for the distributed arc detection platform.

## Overview

The orchestrator manages the lifecycle, health monitoring, and inter-service communication for all microservices in the arc detection system.

## Features

- **Service Registry**: Central registry for service discovery and health monitoring
- **Health Monitoring**: Automated health checks and service restart capabilities  
- **Docker Orchestration**: Docker Compose setup for containerized deployment
- **Configuration Management**: Centralized configuration for all services
- **Monitoring Integration**: Prometheus and Grafana integration for observability
- **Web Dashboard**: Professional web interface with universal styling

## Services Managed

- `data_processor`: Main data processing service (port 5000)
- `transient_viewer`: Transient visualization service (port 5001)
- `segment_visualizer`: Data segment visualization service (port 5002)
- `cache_manager`: Caching and data management service (port 5003)

## Quick Start

### Using Docker Compose
```bash
docker-compose up -d
```

### Manual Orchestration
```bash
python orchestrator.py --config config/orchestrator.json
```

## Web Dashboard

Access the orchestrator dashboard at: http://localhost:8000
- Real-time service status monitoring
- Health check functionality
- Service discovery tools
- Professional styling with universal color palette

## Configuration

Edit `config/orchestrator.json` to customize:
- Service definitions and dependencies
- Startup order and timeouts
- Health check intervals
- Monitoring settings

## Scripts

The `scripts/` directory contains startup and shutdown scripts for individual services:
- `start_*.sh`: Service startup scripts
- `shutdown_*.sh`: Service shutdown scripts

## Monitoring

- **Web Dashboard**: http://localhost:8000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Orchestrator Status**: http://localhost:8000/api/services
