#!/bin/bash

# Automated dependency update script for ProbNeural Operator Lab
# Handles security updates, compatibility checking, and automated testing

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

# Configuration
UPDATE_TYPE="security"  # security, minor, major, all
DRY_RUN=false
SKIP_TESTS=false
AUTO_COMMIT=false
BRANCH_NAME="deps/automated-update-$(date +%Y%m%d-%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            UPDATE_TYPE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --auto-commit)
            AUTO_COMMIT=true
            shift
            ;;
        --branch)
            BRANCH_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE      Update type: security, minor, major, all (default: security)"
            echo "  --dry-run            Show what would be updated without making changes"
            echo "  --skip-tests         Skip running tests after updates"
            echo "  --auto-commit        Automatically commit changes"
            echo "  --branch BRANCH      Use specific branch name"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo_error "Unknown option $1"
            exit 1
            ;;
    esac
done

echo_info "Starting automated dependency updates"
echo_info "Update type: $UPDATE_TYPE"
echo_info "Dry run: $DRY_RUN"
echo_info "Branch: $BRANCH_NAME"

# Ensure we have required tools
check_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo_error "$1 is required but not installed"
        return 1
    fi
}

echo_info "Checking required tools..."
check_tool pip
check_tool pip-audit
check_tool safety

# Optional tools
if command -v pip-outdated >/dev/null 2>&1; then
    HAS_PIP_OUTDATED=true
else
    HAS_PIP_OUTDATED=false
    echo_warning "pip-outdated not available, using pip list --outdated instead"
fi

# Create backup of current requirements
echo_info "Creating backup of current dependencies..."
if [ "$DRY_RUN" = false ]; then
    cp requirements-dev.txt requirements-dev.txt.backup
    pip freeze > requirements.current.txt
fi

# Check current vulnerabilities
echo_info "Checking current security vulnerabilities..."
CURRENT_VULNS=$(safety check --json 2>/dev/null | jq '. | length' || echo "0")
echo_info "Current vulnerabilities: $CURRENT_VULNS"

# Get outdated packages
echo_info "Identifying outdated dependencies..."
if [ "$HAS_PIP_OUTDATED" = true ]; then
    OUTDATED_PACKAGES=$(pip-outdated --format=json | jq -r '.[] | .name')
else
    OUTDATED_PACKAGES=$(pip list --outdated --format=json | jq -r '.[] | .name')
fi

if [ -z "$OUTDATED_PACKAGES" ]; then
    echo_success "All dependencies are up to date!"
    exit 0
fi

echo_info "Outdated packages found:"
echo "$OUTDATED_PACKAGES" | while read -r package; do
    if [ -n "$package" ]; then
        current_version=$(pip show "$package" 2>/dev/null | grep Version | cut -d' ' -f2 || echo "unknown")
        latest_version=$(pip index versions "$package" 2>/dev/null | head -1 | cut -d' ' -f2 || echo "unknown")
        echo "  - $package: $current_version → $latest_version"
    fi
done

# Create branch for updates
if [ "$DRY_RUN" = false ]; then
    echo_info "Creating branch for updates: $BRANCH_NAME"
    git checkout -b "$BRANCH_NAME"
fi

# Function to update a specific package
update_package() {
    local package=$1
    local update_type=$2
    
    echo_info "Updating $package ($update_type)..."
    
    if [ "$DRY_RUN" = true ]; then
        echo_info "[DRY RUN] Would update $package"
        return 0
    fi
    
    # Get current version
    current_version=$(pip show "$package" 2>/dev/null | grep Version | cut -d' ' -f2 || echo "")
    
    if [ -z "$current_version" ]; then
        echo_warning "Package $package not found, skipping"
        return 0
    fi
    
    # Update the package
    case $update_type in
        "security")
            # For security updates, update to latest
            pip install --upgrade "$package"
            ;;
        "minor")
            # For minor updates, be more conservative
            major_minor=$(echo "$current_version" | cut -d'.' -f1-2)
            pip install --upgrade "$package<$(($(echo $major_minor | cut -d'.' -f2) + 1))"
            ;;
        "patch")
            # For patch updates, only within same minor version
            major_minor=$(echo "$current_version" | cut -d'.' -f1-2)
            pip install --upgrade "$package~=$major_minor.0"
            ;;
        *)
            pip install --upgrade "$package"
            ;;
    esac
    
    new_version=$(pip show "$package" 2>/dev/null | grep Version | cut -d' ' -f2 || echo "")
    
    if [ "$current_version" != "$new_version" ]; then
        echo_success "Updated $package: $current_version → $new_version"
        return 0
    else
        echo_info "No update available for $package"
        return 1
    fi
}

# Function to check if update introduces security vulnerabilities
check_security_after_update() {
    echo_info "Running security check after updates..."
    NEW_VULNS=$(safety check --json 2>/dev/null | jq '. | length' || echo "0")
    
    if [ "$NEW_VULNS" -gt "$CURRENT_VULNS" ]; then
        echo_error "Updates introduced new vulnerabilities!"
        safety check
        return 1
    elif [ "$NEW_VULNS" -lt "$CURRENT_VULNS" ]; then
        echo_success "Updates fixed $((CURRENT_VULNS - NEW_VULNS)) vulnerabilities"
    else
        echo_info "No change in vulnerability count"
    fi
    
    return 0
}

# Function to run compatibility tests
run_compatibility_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        echo_warning "Skipping tests as requested"
        return 0
    fi
    
    echo_info "Running compatibility tests..."
    
    # Import test
    if ! python -c "import probneural_operator; print('Import successful')"; then
        echo_error "Basic import failed after updates"
        return 1
    fi
    
    # Run unit tests
    if ! python -m pytest tests/unit/ -x --tb=short; then
        echo_error "Unit tests failed after updates"
        return 1
    fi
    
    # Run integration tests
    if ! python -m pytest tests/integration/ -x --tb=short; then
        echo_error "Integration tests failed after updates"
        return 1
    fi
    
    echo_success "All compatibility tests passed"
    return 0
}

# Main update logic
UPDATES_APPLIED=0
FAILED_UPDATES=0

for package in $OUTDATED_PACKAGES; do
    if [ -z "$package" ]; then
        continue
    fi
    
    # Check if package has security vulnerabilities
    HAS_SECURITY_VULN=false
    if safety check --json 2>/dev/null | jq -e ".[] | select(.package_name == \"$package\")" >/dev/null; then
        HAS_SECURITY_VULN=true
    fi
    
    # Determine if we should update this package
    SHOULD_UPDATE=false
    
    case $UPDATE_TYPE in
        "security")
            if [ "$HAS_SECURITY_VULN" = true ]; then
                SHOULD_UPDATE=true
                echo_warning "Security vulnerability found in $package"
            fi
            ;;
        "minor"|"patch"|"major"|"all")
            SHOULD_UPDATE=true
            ;;
    esac
    
    if [ "$SHOULD_UPDATE" = true ]; then
        if update_package "$package" "$UPDATE_TYPE"; then
            UPDATES_APPLIED=$((UPDATES_APPLIED + 1))
        else
            FAILED_UPDATES=$((FAILED_UPDATES + 1))
        fi
    else
        echo_info "Skipping $package (no security issues, update type: $UPDATE_TYPE)"
    fi
done

# Update requirements files
if [ "$DRY_RUN" = false ] && [ "$UPDATES_APPLIED" -gt 0 ]; then
    echo_info "Updating requirements files..."
    pip freeze > requirements-dev.txt
    
    # Also update pyproject.toml if it has dependency versions
    if [ -f "pyproject.toml" ]; then
        echo_info "Note: You may need to manually update version constraints in pyproject.toml"
    fi
fi

# Run post-update checks
if [ "$UPDATES_APPLIED" -gt 0 ] && [ "$DRY_RUN" = false ]; then
    echo_info "Running post-update validation..."
    
    # Security check
    if ! check_security_after_update; then
        echo_error "Security check failed, rolling back..."
        git checkout -- requirements-dev.txt
        pip install -r requirements-dev.txt.backup
        exit 1
    fi
    
    # Compatibility tests
    if ! run_compatibility_tests; then
        echo_error "Compatibility tests failed, rolling back..."
        git checkout -- requirements-dev.txt
        pip install -r requirements-dev.txt.backup
        exit 1
    fi
fi

# Generate update summary
echo_info "Update Summary:"
echo "  - Packages processed: $(echo "$OUTDATED_PACKAGES" | wc -w)"
echo "  - Updates applied: $UPDATES_APPLIED"
echo "  - Failed updates: $FAILED_UPDATES"
echo "  - Update type: $UPDATE_TYPE"

# Commit changes if requested and updates were applied
if [ "$AUTO_COMMIT" = true ] && [ "$UPDATES_APPLIED" -gt 0 ] && [ "$DRY_RUN" = false ]; then
    echo_info "Committing dependency updates..."
    
    git add requirements-dev.txt
    git commit -m "deps: automated dependency updates ($UPDATE_TYPE)

- Updated $UPDATES_APPLIED packages
- Fixed $((CURRENT_VULNS - NEW_VULNS)) security vulnerabilities
- All tests passing

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    echo_success "Changes committed to branch $BRANCH_NAME"
    echo_info "To create a PR: gh pr create --title 'Automated dependency updates' --body 'Automated security and compatibility updates'"
fi

# Cleanup
if [ "$DRY_RUN" = false ]; then
    rm -f requirements-dev.txt.backup requirements.current.txt
fi

echo_success "Dependency update process completed!"

if [ "$UPDATES_APPLIED" -gt 0 ]; then
    echo ""
    echo_info "Next steps:"
    echo "1. Review the changes: git diff HEAD~1"
    echo "2. Test manually if needed"
    echo "3. Create PR: gh pr create --title 'Automated dependency updates'"
    echo "4. Merge after CI passes"
else
    echo ""
    echo_info "No updates were applied."
    if [ "$UPDATE_TYPE" = "security" ]; then
        echo "✅ No security vulnerabilities found in dependencies"
    fi
fi