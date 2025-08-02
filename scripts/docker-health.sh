#!/bin/bash

# Docker health check script for ProbNeural Operator Lab
# Monitors container health and provides diagnostics

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONTAINER_NAME=""
CHECK_LOGS=false
FOLLOW_LOGS=false
WATCH_MODE=false
INTERVAL=5

print_usage() {
    echo "Usage: $0 [OPTIONS] [CONTAINER_NAME]"
    echo ""
    echo "Options:"
    echo "  -c, --container NAME   Check specific container"
    echo "  -l, --logs            Show container logs"
    echo "  -f, --follow          Follow log output"
    echo "  -w, --watch           Watch mode (continuous monitoring)"
    echo "  -i, --interval SEC    Watch interval in seconds (default: 5)"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Check all probneural containers"
    echo "  $0 -c probneural-dev                # Check specific container"
    echo "  $0 -l -c probneural-dev             # Show logs for container"
    echo "  $0 -w -i 10                         # Watch all containers every 10 seconds"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--container)
            CONTAINER_NAME="$2"
            shift 2
            ;;
        -l|--logs)
            CHECK_LOGS=true
            shift
            ;;
        -f|--follow)
            FOLLOW_LOGS=true
            CHECK_LOGS=true
            shift
            ;;
        -w|--watch)
            WATCH_MODE=true
            shift
            ;;
        -i|--interval)
            INTERVAL="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            if [[ -z "$CONTAINER_NAME" ]]; then
                CONTAINER_NAME="$1"
            else
                echo "Unknown option: $1"
                print_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Function to check Docker daemon
check_docker_daemon() {
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Docker daemon is not running${NC}"
        echo "Please start Docker daemon and try again."
        exit 1
    fi
}

# Function to get container status
get_container_status() {
    local container=$1
    docker inspect "$container" --format '{{.State.Status}}' 2>/dev/null || echo "not_found"
}

# Function to get container health
get_container_health() {
    local container=$1
    docker inspect "$container" --format '{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck"
}

# Function to check container resources
check_container_resources() {
    local container=$1
    
    echo -e "${BLUE}Resource Usage for $container:${NC}"
    
    # CPU and Memory usage
    local stats=$(docker stats "$container" --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" 2>/dev/null)
    if [[ $? -eq 0 ]]; then
        echo "$stats"
    else
        echo "  Unable to get resource stats"
    fi
    
    # Port mappings
    echo -e "\n${BLUE}Port Mappings:${NC}"
    docker port "$container" 2>/dev/null || echo "  No port mappings"
    
    # Volume mounts
    echo -e "\n${BLUE}Volume Mounts:${NC}"
    docker inspect "$container" --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Type}}){{"\n"}}{{end}}' 2>/dev/null || echo "  No volume mounts"
}

# Function to check single container
check_container() {
    local container=$1
    local status=$(get_container_status "$container")
    local health=$(get_container_health "$container")
    
    echo -e "${BLUE}Container: $container${NC}"
    echo "========================================="
    
    case $status in
        "running")
            echo -e "Status: ${GREEN}✅ Running${NC}"
            ;;
        "exited")
            echo -e "Status: ${RED}❌ Exited${NC}"
            ;;
        "paused")
            echo -e "Status: ${YELLOW}⏸️  Paused${NC}"
            ;;
        "restarting")
            echo -e "Status: ${YELLOW}🔄 Restarting${NC}"
            ;;
        "not_found")
            echo -e "Status: ${RED}❌ Not Found${NC}"
            return 1
            ;;
        *)
            echo -e "Status: ${YELLOW}⚠️  $status${NC}"
            ;;
    esac
    
    # Health check status
    case $health in
        "healthy")
            echo -e "Health: ${GREEN}✅ Healthy${NC}"
            ;;
        "unhealthy")
            echo -e "Health: ${RED}❌ Unhealthy${NC}"
            ;;
        "starting")
            echo -e "Health: ${YELLOW}🔄 Starting${NC}"
            ;;
        "no_healthcheck")
            echo -e "Health: ${BLUE}ℹ️  No healthcheck${NC}"
            ;;
        *)
            echo -e "Health: ${YELLOW}⚠️  $health${NC}"
            ;;
    esac
    
    # Show container details if running
    if [[ "$status" == "running" ]]; then
        echo -e "\n${BLUE}Container Details:${NC}"
        docker inspect "$container" --format '
Image: {{.Config.Image}}
Created: {{.Created}}
Started: {{.State.StartedAt}}
Uptime: {{.State.StartedAt}}
RestartCount: {{.RestartCount}}
' 2>/dev/null
        
        check_container_resources "$container"
    fi
    
    # Show recent logs if container has issues
    if [[ "$status" != "running" || "$health" == "unhealthy" ]]; then
        echo -e "\n${YELLOW}Recent logs (last 10 lines):${NC}"
        docker logs --tail 10 "$container" 2>&1 || echo "  Unable to get logs"
    fi
}

# Function to show logs
show_logs() {
    local container=$1
    
    echo -e "${BLUE}Logs for $container:${NC}"
    echo "=============================="
    
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        docker logs -f "$container"
    else
        docker logs --tail 50 "$container"
    fi
}

# Function to monitor all probneural containers
monitor_containers() {
    echo -e "${BLUE}ProbNeural Operator Lab - Container Health Monitor${NC}"
    echo -e "${BLUE}=================================================${NC}"
    echo "Timestamp: $(date)"
    echo ""
    
    # Find all probneural related containers
    local containers=$(docker ps -a --filter "name=probneural" --format "{{.Names}}" 2>/dev/null)
    
    if [[ -z "$containers" ]]; then
        echo -e "${YELLOW}No probneural containers found${NC}"
        echo ""
        echo "Available containers:"
        docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
        return
    fi
    
    echo "Found containers: $containers"
    echo ""
    
    for container in $containers; do
        check_container "$container"
        echo ""
    done
    
    # Show Docker system resources
    echo -e "${BLUE}Docker System Resources:${NC}"
    echo "=========================="
    docker system df
    echo ""
    
    # Show running services from docker-compose
    if [[ -f "docker-compose.yml" ]]; then
        echo -e "${BLUE}Docker Compose Services:${NC}"
        echo "========================"
        docker-compose ps 2>/dev/null || echo "  Unable to get compose status"
    fi
}

# Main execution
main() {
    check_docker_daemon
    
    if [[ "$WATCH_MODE" == "true" ]]; then
        echo -e "${GREEN}Starting watch mode (interval: ${INTERVAL}s, press Ctrl+C to stop)${NC}"
        echo ""
        
        while true; do
            clear
            monitor_containers
            sleep "$INTERVAL"
        done
    elif [[ -n "$CONTAINER_NAME" ]]; then
        if [[ "$CHECK_LOGS" == "true" ]]; then
            show_logs "$CONTAINER_NAME"
        else
            check_container "$CONTAINER_NAME"
        fi
    else
        monitor_containers
    fi
}

# Handle Ctrl+C gracefully in watch mode
trap 'echo -e "\n${GREEN}Health monitoring stopped${NC}"; exit 0' INT

main "$@"