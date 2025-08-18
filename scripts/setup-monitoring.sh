#!/bin/bash

# Monitoring and Observability Setup Script
# Sets up comprehensive monitoring stack for ProbNeural Operator Lab

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Default values
INSTALL_PROMETHEUS=true
INSTALL_GRAFANA=true
INSTALL_ALERTMANAGER=true
INSTALL_LOKI=true
INSTALL_JAEGER=false
ENVIRONMENT="development"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-prometheus)
            INSTALL_PROMETHEUS=false
            shift
            ;;
        --no-grafana)
            INSTALL_GRAFANA=false
            shift
            ;;
        --no-alertmanager)
            INSTALL_ALERTMANAGER=false
            shift
            ;;
        --no-loki)
            INSTALL_LOKI=false
            shift
            ;;
        --with-jaeger)
            INSTALL_JAEGER=true
            shift
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-prometheus      Skip Prometheus installation"
            echo "  --no-grafana         Skip Grafana installation"
            echo "  --no-alertmanager    Skip Alertmanager installation"
            echo "  --no-loki            Skip Loki installation"
            echo "  --with-jaeger        Install Jaeger for distributed tracing"
            echo "  --env ENVIRONMENT    Environment: development, production (default: development)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo_error "Unknown option $1"
            exit 1
            ;;
    esac
done

echo_info "Setting up monitoring stack for ProbNeural Operator Lab"
echo_info "Environment: $ENVIRONMENT"

# Create monitoring directory structure
echo_info "Creating monitoring directories..."
mkdir -p monitoring/data/{prometheus,grafana,alertmanager,loki}
mkdir -p monitoring/configs
mkdir -p logs
mkdir -p dashboards

# Create docker-compose for monitoring stack
echo_info "Creating monitoring docker-compose configuration..."

cat > monitoring/docker-compose.monitoring.yml << EOF
version: '3.8'

services:
EOF

# Add Prometheus if requested
if [ "$INSTALL_PROMETHEUS" = true ]; then
    echo_info "Adding Prometheus configuration..."
    cat >> monitoring/docker-compose.monitoring.yml << EOF
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: probneural-prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts.yml:/etc/prometheus/alerts.yml
      - ./data/prometheus:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring
    restart: unless-stopped

EOF
fi

# Add Grafana if requested
if [ "$INSTALL_GRAFANA" = true ]; then
    echo_info "Adding Grafana configuration..."
    cat >> monitoring/docker-compose.monitoring.yml << EOF
  grafana:
    image: grafana/grafana:10.0.0
    container_name: probneural-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - ./data/grafana:/var/lib/grafana
      - ./grafana-dashboards.json:/var/lib/grafana/dashboards/dashboards.json
    ports:
      - "3000:3000"
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - prometheus

EOF
fi

# Add Alertmanager if requested
if [ "$INSTALL_ALERTMANAGER" = true ]; then
    echo_info "Adding Alertmanager configuration..."
    cat >> monitoring/docker-compose.monitoring.yml << EOF
  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: probneural-alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - ./data/alertmanager:/alertmanager
    ports:
      - "9093:9093"
    networks:
      - monitoring
    restart: unless-stopped

EOF
fi

# Add Loki if requested
if [ "$INSTALL_LOKI" = true ]; then
    echo_info "Adding Loki configuration..."
    cat >> monitoring/docker-compose.monitoring.yml << EOF
  loki:
    image: grafana/loki:2.8.0
    container_name: probneural-loki
    ports:
      - "3100:3100"
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./data/loki:/loki
    networks:
      - monitoring
    restart: unless-stopped

  promtail:
    image: grafana/promtail:2.8.0
    container_name: probneural-promtail
    volumes:
      - ../logs:/var/log/probneural:ro
      - ./promtail-config.yml:/etc/promtail/config.yml
    command: -config.file=/etc/promtail/config.yml
    networks:
      - monitoring
    restart: unless-stopped
    depends_on:
      - loki

EOF

    # Create Promtail configuration
    cat > monitoring/promtail-config.yml << EOF
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
- job_name: probneural-logs
  static_configs:
  - targets:
      - localhost
    labels:
      job: probneural-logs
      __path__: /var/log/probneural/*.log

- job_name: probneural-json-logs
  static_configs:
  - targets:
      - localhost
    labels:
      job: probneural-json-logs
      __path__: /var/log/probneural/*.json
  pipeline_stages:
  - json:
      expressions:
        timestamp: asctime
        level: levelname
        message: message
        logger: name
EOF
fi

# Add Jaeger if requested
if [ "$INSTALL_JAEGER" = true ]; then
    echo_info "Adding Jaeger configuration..."
    cat >> monitoring/docker-compose.monitoring.yml << EOF
  jaeger:
    image: jaegertracing/all-in-one:1.46
    container_name: probneural-jaeger
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # HTTP collector
      - "9411:9411"    # Zipkin collector
    networks:
      - monitoring
    restart: unless-stopped

EOF
fi

# Add Node Exporter for system metrics
cat >> monitoring/docker-compose.monitoring.yml << EOF
  node-exporter:
    image: prom/node-exporter:v1.6.0
    container_name: probneural-node-exporter
    command:
      - '--path.rootfs=/host'
    volumes:
      - '/:/host:ro,rslave'
    ports:
      - "9100:9100"
    networks:
      - monitoring
    restart: unless-stopped

EOF

# Add cAdvisor for container metrics
cat >> monitoring/docker-compose.monitoring.yml << EOF
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: probneural-cadvisor
    privileged: true
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - "8080:8080"
    networks:
      - monitoring
    restart: unless-stopped

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
  loki-data:
  alertmanager-data:
EOF

# Create Grafana provisioning configurations
echo_info "Setting up Grafana provisioning..."
mkdir -p monitoring/grafana/provisioning/{datasources,dashboards}

cat > monitoring/grafana/provisioning/datasources/datasources.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

if [ "$INSTALL_LOKI" = true ]; then
    cat >> monitoring/grafana/provisioning/datasources/datasources.yml << EOF
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true
EOF
fi

if [ "$INSTALL_JAEGER" = true ]; then
    cat >> monitoring/grafana/provisioning/datasources/datasources.yml << EOF
  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
EOF
fi

cat > monitoring/grafana/provisioning/dashboards/dashboards.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create environment-specific configurations
if [ "$ENVIRONMENT" = "production" ]; then
    echo_info "Applying production configurations..."
    
    # Update Grafana security
    sed -i 's/GF_SECURITY_ADMIN_PASSWORD=admin123/GF_SECURITY_ADMIN_PASSWORD=${GF_ADMIN_PASSWORD}/' monitoring/docker-compose.monitoring.yml
    
    # Add resource limits
    echo_info "Adding resource limits for production..."
    # This would typically be done with docker-compose overrides
fi

# Create monitoring startup script
cat > monitoring/start-monitoring.sh << 'EOF'
#!/bin/bash
set -e

echo "Starting ProbNeural Operator Lab monitoring stack..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not running"
    exit 1
fi

# Create data directories with correct permissions
mkdir -p data/{prometheus,grafana,alertmanager,loki}
chmod 777 data/grafana  # Grafana needs write access

# Start the monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

echo "Monitoring stack started successfully!"
echo ""
echo "Access URLs:"
echo "• Prometheus: http://localhost:9090"
echo "• Grafana: http://localhost:3000 (admin/admin123)"
echo "• Alertmanager: http://localhost:9093"
echo "• Loki: http://localhost:3100"
if [ -f docker-compose.monitoring.yml ] && grep -q "jaeger" docker-compose.monitoring.yml; then
    echo "• Jaeger: http://localhost:16686"
fi
echo ""
echo "System metrics:"
echo "• Node Exporter: http://localhost:9100"
echo "• cAdvisor: http://localhost:8080"
EOF

chmod +x monitoring/start-monitoring.sh

# Create monitoring stop script
cat > monitoring/stop-monitoring.sh << 'EOF'
#!/bin/bash
set -e

echo "Stopping ProbNeural Operator Lab monitoring stack..."
docker-compose -f docker-compose.monitoring.yml down
echo "Monitoring stack stopped successfully!"
EOF

chmod +x monitoring/stop-monitoring.sh

# Create health check script
cat > monitoring/health-check.sh << 'EOF'
#!/bin/bash

# Health check script for monitoring stack

check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $service_name..."
    
    if response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null); then
        if [ "$response" = "$expected_status" ]; then
            echo " ✅ OK ($response)"
        else
            echo " ❌ FAIL ($response)"
            return 1
        fi
    else
        echo " ❌ UNREACHABLE"
        return 1
    fi
}

echo "ProbNeural Operator Lab - Monitoring Health Check"
echo "================================================"

all_healthy=true

check_service "Prometheus" "http://localhost:9090/-/healthy" || all_healthy=false
check_service "Grafana" "http://localhost:3000/api/health" || all_healthy=false
check_service "Alertmanager" "http://localhost:9093/-/healthy" || all_healthy=false
check_service "Loki" "http://localhost:3100/ready" || all_healthy=false
check_service "Node Exporter" "http://localhost:9100/metrics" || all_healthy=false
check_service "cAdvisor" "http://localhost:8080/metrics" || all_healthy=false

if grep -q "jaeger" docker-compose.monitoring.yml; then
    check_service "Jaeger" "http://localhost:16686/" || all_healthy=false
fi

echo ""
if [ "$all_healthy" = true ]; then
    echo "🎉 All monitoring services are healthy!"
    exit 0
else
    echo "⚠️  Some monitoring services are not healthy"
    exit 1
fi
EOF

chmod +x monitoring/health-check.sh

echo_success "Monitoring setup completed!"

echo ""
echo_info "Monitoring Stack Summary:"
echo "• Prometheus: $INSTALL_PROMETHEUS"
echo "• Grafana: $INSTALL_GRAFANA" 
echo "• Alertmanager: $INSTALL_ALERTMANAGER"
echo "• Loki: $INSTALL_LOKI"
echo "• Jaeger: $INSTALL_JAEGER"
echo "• Environment: $ENVIRONMENT"

echo ""
echo_info "Next steps:"
echo "1. Review and customize monitoring/alertmanager.yml with your notification settings"
echo "2. Start monitoring stack: cd monitoring && ./start-monitoring.sh"
echo "3. Check health: cd monitoring && ./health-check.sh"
echo "4. Import dashboards into Grafana"
echo "5. Configure alerts for your specific use case"

echo ""
echo_success "🔍 Monitoring and observability setup complete!"