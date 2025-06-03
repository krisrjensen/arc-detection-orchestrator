#!/usr/bin/env python3
"""
Web Interface for Service Registry Dashboard

Provides a web-based dashboard for monitoring and managing services
with universal styling applied.
"""

from flask import Flask, render_template, jsonify, request
import json
import logging
from datetime import datetime
from service_registry import ServiceRegistry, ServiceInfo

# Universal color palette from Style Manager
UNIVERSAL_COLORS = {
    'primary': '#1f77b4',
    'background': '#ffffff',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'warning': '#d62728',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40'
}

class RegistryWebInterface:
    def __init__(self, registry: ServiceRegistry, port: int = 4080):
        self.app = Flask(__name__)
        self.registry = registry
        self.port = port
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html', colors=UNIVERSAL_COLORS)
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'orchestrator'
            })
        
        @self.app.route('/api/services')
        def get_services():
            """Get all registered services"""
            services = self.registry.get_all_services()
            return jsonify({
                'services': {
                    name: {
                        'name': service.name,
                        'host': service.host,
                        'port': service.port,
                        'status': service.status,
                        'last_heartbeat': service.last_heartbeat.isoformat() if service.last_heartbeat else None,
                        'health_endpoint': service.health_endpoint,
                        'metadata': service.metadata
                    }
                    for name, service in services.items()
                },
                'total_services': len(services),
                'healthy_services': len([s for s in services.values() if s.status == 'healthy'])
            })
        
        @self.app.route('/api/services/<service_name>/health')
        def check_service_health(service_name):
            """Check health of a specific service"""
            service = self.registry.get_service(service_name)
            if not service:
                return jsonify({'error': 'Service not found'}), 404
            
            is_healthy = self.registry._verify_service_health(service)
            return jsonify({
                'service': service_name,
                'healthy': is_healthy,
                'status': service.status,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/api/discover')
        def discover_services():
            """Discover services on common ports"""
            discovered = self.registry.discover_services()
            return jsonify({
                'discovered_services': [
                    {
                        'name': service.name,
                        'host': service.host,
                        'port': service.port,
                        'health_endpoint': service.health_endpoint
                    }
                    for service in discovered
                ],
                'count': len(discovered)
            })
        
        @self.app.route('/api/services', methods=['POST'])
        def register_service():
            """Register a new service"""
            try:
                data = request.get_json()
                service_info = ServiceInfo(
                    name=data['name'],
                    host=data.get('host', 'localhost'),
                    port=data['port'],
                    health_endpoint=data.get('health_endpoint', '/health'),
                    metadata=data.get('metadata', {})
                )
                
                success = self.registry.register_service(service_info)
                return jsonify({
                    'success': success,
                    'message': f"Service {data['name']} {'registered' if success else 'registration failed'}"
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 400
    
    def create_dashboard_template(self):
        """Create the HTML dashboard template"""
        template_dir = 'templates'
        import os
        os.makedirs(template_dir, exist_ok=True)
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Arc Detection Service Registry</title>
    <style>
        :root {{
            --primary-color: {UNIVERSAL_COLORS['primary']};
            --background-color: {UNIVERSAL_COLORS['background']};
            --secondary-color: {UNIVERSAL_COLORS['secondary']};
            --success-color: {UNIVERSAL_COLORS['success']};
            --warning-color: {UNIVERSAL_COLORS['warning']};
            --info-color: {UNIVERSAL_COLORS['info']};
            --light-color: {UNIVERSAL_COLORS['light']};
            --dark-color: {UNIVERSAL_COLORS['dark']};
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--background-color);
            color: var(--dark-color);
            line-height: 1.6;
        }}
        
        .header {{
            background-color: var(--primary-color);
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--light-color);
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid var(--primary-color);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .stat-label {{
            color: #666;
            margin-top: 0.5rem;
        }}
        
        .services-table {{
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .table-header {{
            background: var(--primary-color);
            color: white;
            padding: 1rem 2rem;
            font-weight: 600;
        }}
        
        .service-row {{
            display: grid;
            grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr;
            padding: 1rem 2rem;
            border-bottom: 1px solid #eee;
            align-items: center;
        }}
        
        .service-row:hover {{
            background-color: var(--light-color);
        }}
        
        .status-badge {{
            padding: 0.25rem 0.75rem;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 500;
            text-align: center;
        }}
        
        .status-healthy {{
            background-color: var(--success-color);
            color: white;
        }}
        
        .status-unhealthy {{
            background-color: var(--warning-color);
            color: white;
        }}
        
        .status-unknown {{
            background-color: #6c757d;
            color: white;
        }}
        
        .btn {{
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.875rem;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }}
        
        .btn-primary {{
            background-color: var(--primary-color);
            color: white;
        }}
        
        .btn-success {{
            background-color: var(--success-color);
            color: white;
        }}
        
        .btn:hover {{
            opacity: 0.9;
        }}
        
        .actions {{
            margin: 1rem 2rem;
        }}
        
        .loading {{
            text-align: center;
            padding: 2rem;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Arc Detection Service Registry</h1>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-services">0</div>
                <div class="stat-label">Total Services</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="healthy-services">0</div>
                <div class="stat-label">Healthy Services</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="uptime">100%</div>
                <div class="stat-label">System Uptime</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="last-update">Just now</div>
                <div class="stat-label">Last Update</div>
            </div>
        </div>
        
        <div class="services-table">
            <div class="table-header">
                Registered Services
            </div>
            <div class="actions">
                <button class="btn btn-primary" onclick="refreshServices()">🔄 Refresh</button>
                <button class="btn btn-success" onclick="discoverServices()">🔍 Discover Services</button>
            </div>
            <div class="service-row" style="font-weight: bold; background: var(--light-color);">
                <div>Service Name</div>
                <div>Host</div>
                <div>Port</div>
                <div>Status</div>
                <div>Last Heartbeat</div>
                <div>Actions</div>
            </div>
            <div id="services-list">
                <div class="loading">Loading services...</div>
            </div>
        </div>
    </div>
    
    <script>
        async function fetchServices() {{
            try {{
                const response = await fetch('/api/services');
                const data = await response.json();
                
                document.getElementById('total-services').textContent = data.total_services;
                document.getElementById('healthy-services').textContent = data.healthy_services;
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
                
                const servicesList = document.getElementById('services-list');
                servicesList.innerHTML = '';
                
                Object.values(data.services).forEach(service => {{
                    const row = document.createElement('div');
                    row.className = 'service-row';
                    
                    const lastHeartbeat = service.last_heartbeat ? 
                        new Date(service.last_heartbeat).toLocaleTimeString() : 'Never';
                    
                    row.innerHTML = `
                        <div>${{service.name}}</div>
                        <div>${{service.host}}</div>
                        <div>${{service.port}}</div>
                        <div><span class="status-badge status-${{service.status}}">${{service.status.toUpperCase()}}</span></div>
                        <div>${{lastHeartbeat}}</div>
                        <div>
                            <button class="btn btn-primary" onclick="checkHealth('${{service.name}}')">Check Health</button>
                        </div>
                    `;
                    servicesList.appendChild(row);
                }});
                
                if (Object.keys(data.services).length === 0) {{
                    servicesList.innerHTML = '<div class="loading">No services registered</div>';
                }}
            }} catch (error) {{
                console.error('Failed to fetch services:', error);
                document.getElementById('services-list').innerHTML = 
                    '<div class="loading" style="color: var(--warning-color);">Failed to load services</div>';
            }}
        }}
        
        async function checkHealth(serviceName) {{
            try {{
                const response = await fetch(`/api/services/${{serviceName}}/health`);
                const data = await response.json();
                alert(`Service ${{serviceName}} health: ${{data.healthy ? 'Healthy' : 'Unhealthy'}}`);
                refreshServices();
            }} catch (error) {{
                alert(`Failed to check health for ${{serviceName}}`);
            }}
        }}
        
        async function discoverServices() {{
            try {{
                const response = await fetch('/api/discover');
                const data = await response.json();
                alert(`Discovered ${{data.count}} services`);
                refreshServices();
            }} catch (error) {{
                alert('Failed to discover services');
            }}
        }}
        
        function refreshServices() {{
            fetchServices();
        }}
        
        // Initial load and auto-refresh
        fetchServices();
        setInterval(fetchServices, 10000); // Refresh every 10 seconds
    </script>
</body>
</html>"""
        
        with open(f'{template_dir}/dashboard.html', 'w') as f:
            f.write(html_content)
    
    def run(self, debug: bool = False):
        """Run the web interface"""
        self.create_dashboard_template()
        self.logger.info(f"Starting web interface on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    registry = ServiceRegistry()
    web_interface = RegistryWebInterface(registry)
    web_interface.run(debug=True)