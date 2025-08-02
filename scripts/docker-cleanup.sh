#!/bin/bash

# Docker cleanup script for ProbNeural Operator Lab
# Removes unused images, containers, and volumes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Docker Cleanup for ProbNeural Operator Lab${NC}"
echo -e "${BLUE}==========================================${NC}"

# Parse command line arguments
FORCE=false
VOLUMES=false
ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force|-f)
            FORCE=true
            shift
            ;;
        --volumes|-v)
            VOLUMES=true
            shift
            ;;
        --all|-a)
            ALL=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -f, --force    Force removal without confirmation"
            echo "  -v, --volumes  Also remove volumes"
            echo "  -a, --all      Remove everything (containers, images, volumes, networks)"
            echo "  -h, --help     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Function to ask for confirmation
confirm() {
    if [[ "$FORCE" == "true" ]]; then
        return 0
    fi
    
    read -p "$1 (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        return 0
    else
        return 1
    fi
}

echo "Current Docker usage:"
echo "Containers: $(docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Size}}')"
echo ""
echo "Images: $(docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}')"
echo ""

if [[ "$ALL" == "true" ]]; then
    echo -e "${YELLOW}WARNING: This will remove ALL Docker containers, images, volumes, and networks!${NC}"
    if confirm "Are you sure you want to proceed?"; then
        echo -e "${BLUE}Stopping all containers...${NC}"
        docker stop $(docker ps -aq) 2>/dev/null || true
        
        echo -e "${BLUE}Removing all containers...${NC}"
        docker rm $(docker ps -aq) 2>/dev/null || true
        
        echo -e "${BLUE}Removing all images...${NC}"
        docker rmi $(docker images -aq) 2>/dev/null || true
        
        echo -e "${BLUE}Removing all volumes...${NC}"
        docker volume rm $(docker volume ls -q) 2>/dev/null || true
        
        echo -e "${BLUE}Removing all networks...${NC}"
        docker network rm $(docker network ls -q) 2>/dev/null || true
        
        echo -e "${GREEN}✅ Complete cleanup done!${NC}"
        exit 0
    else
        echo "Cleanup cancelled."
        exit 0
    fi
fi

# Remove stopped containers
stopped_containers=$(docker ps -aq --filter "status=exited")
if [[ -n "$stopped_containers" ]]; then
    echo -e "${YELLOW}Found stopped containers:${NC}"
    docker ps -a --filter "status=exited" --format 'table {{.Names}}\t{{.Status}}\t{{.Size}}'
    
    if confirm "Remove stopped containers?"; then
        docker rm $stopped_containers
        echo -e "${GREEN}✅ Removed stopped containers${NC}"
    fi
else
    echo -e "${GREEN}No stopped containers to remove${NC}"
fi

# Remove dangling images
dangling_images=$(docker images -f "dangling=true" -q)
if [[ -n "$dangling_images" ]]; then
    echo -e "${YELLOW}Found dangling images:${NC}"
    docker images -f "dangling=true"
    
    if confirm "Remove dangling images?"; then
        docker rmi $dangling_images
        echo -e "${GREEN}✅ Removed dangling images${NC}"
    fi
else
    echo -e "${GREEN}No dangling images to remove${NC}"
fi

# Remove unused images (not just dangling)
unused_images=$(docker images --filter "label!=keep" --format "{{.ID}}" | head -10)
if [[ -n "$unused_images" ]]; then
    echo -e "${YELLOW}Found potentially unused images (showing first 10):${NC}"
    docker images --filter "label!=keep" --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' | head -11
    
    if confirm "Remove unused images? (This will only remove images without the 'keep' label)"; then
        echo "$unused_images" | xargs docker rmi 2>/dev/null || true
        echo -e "${GREEN}✅ Removed unused images${NC}"
    fi
fi

# Remove probneural-operator-lab specific images
project_images=$(docker images | grep "probneural-operator-lab" | awk '{print $3}')
if [[ -n "$project_images" ]]; then
    echo -e "${YELLOW}Found project-specific images:${NC}"
    docker images | grep "probneural-operator-lab" || true
    
    if confirm "Remove probneural-operator-lab images?"; then
        echo "$project_images" | xargs docker rmi 2>/dev/null || true
        echo -e "${GREEN}✅ Removed project images${NC}"
    fi
fi

# Remove volumes if requested
if [[ "$VOLUMES" == "true" ]]; then
    unused_volumes=$(docker volume ls -f "dangling=true" -q)
    if [[ -n "$unused_volumes" ]]; then
        echo -e "${YELLOW}Found unused volumes:${NC}"
        docker volume ls -f "dangling=true"
        
        if confirm "Remove unused volumes? (This will permanently delete data)"; then
            docker volume rm $unused_volumes
            echo -e "${GREEN}✅ Removed unused volumes${NC}"
        fi
    else
        echo -e "${GREEN}No unused volumes to remove${NC}"
    fi
fi

# System prune
if confirm "Run docker system prune to clean up remaining unused objects?"; then
    if [[ "$VOLUMES" == "true" ]]; then
        docker system prune -f --volumes
    else
        docker system prune -f
    fi
    echo -e "${GREEN}✅ System prune completed${NC}"
fi

echo ""
echo -e "${GREEN}Cleanup Summary${NC}"
echo "=============="
echo "Containers: $(docker ps -a | wc -l) total"
echo "Images: $(docker images | wc -l) total"
echo "Volumes: $(docker volume ls | wc -l) total"
echo ""

# Show disk space freed
echo "Current Docker disk usage:"
docker system df

echo ""
echo -e "${GREEN}🎉 Cleanup completed!${NC}"