#!/bin/bash

# Release automation script for ProbNeural Operator Lab
# Handles semantic versioning, changelog generation, and automated releases

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
VERSION_TYPE="patch"
DRY_RUN=false
SKIP_TESTS=false
SKIP_BUILD=false
BRANCH="main"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            VERSION_TYPE="$2"
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
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE    Version bump type: major, minor, patch (default: patch)"
            echo "  --dry-run          Show what would be done without making changes"
            echo "  --skip-tests       Skip running tests before release"
            echo "  --skip-build       Skip building packages"
            echo "  --branch BRANCH    Target branch (default: main)"
            echo "  -h, --help         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --type minor    # Bump minor version"
            echo "  $0 --dry-run       # Preview release without making changes"
            exit 0
            ;;
        *)
            echo_error "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate version type
case $VERSION_TYPE in
    major|minor|patch)
        ;;
    *)
        echo_error "Invalid version type: $VERSION_TYPE"
        echo "Valid types: major, minor, patch"
        exit 1
        ;;
esac

echo_info "Preparing release for ProbNeural Operator Lab"
echo_info "Version bump type: $VERSION_TYPE"
echo_info "Target branch: $BRANCH"
echo_info "Dry run: $DRY_RUN"

# Pre-flight checks
echo_info "Running pre-flight checks..."

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo_error "Not in a git repository"
    exit 1
fi

# Check if we're on the correct branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo_error "Currently on branch '$CURRENT_BRANCH', expected '$BRANCH'"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo_error "Uncommitted changes found. Please commit or stash them."
    exit 1
fi

# Check if remote is up to date
echo_info "Fetching latest changes..."
git fetch origin

if [ "$(git rev-parse HEAD)" != "$(git rev-parse @{u})" ]; then
    echo_error "Local branch is not up to date with remote. Please pull latest changes."
    exit 1
fi

# Get current version
CURRENT_VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "0.0.0")
echo_info "Current version: $CURRENT_VERSION"

# Calculate new version
calculate_new_version() {
    local current="$1"
    local type="$2"
    
    # Parse semantic version
    if [[ $current =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-(.+))?$ ]]; then
        local major="${BASH_REMATCH[1]}"
        local minor="${BASH_REMATCH[2]}"  
        local patch="${BASH_REMATCH[3]}"
        
        case $type in
            major)
                echo "$((major + 1)).0.0"
                ;;
            minor)
                echo "${major}.$((minor + 1)).0"
                ;;
            patch)
                echo "${major}.${minor}.$((patch + 1))"
                ;;
        esac
    else
        echo_error "Invalid version format: $current"
        exit 1
    fi
}

NEW_VERSION=$(calculate_new_version "$CURRENT_VERSION" "$VERSION_TYPE")
echo_info "New version: $NEW_VERSION"

if [ "$DRY_RUN" = true ]; then
    echo_warning "DRY RUN MODE - No changes will be made"
fi

# Run tests
if [ "$SKIP_TESTS" = false ]; then
    echo_info "Running test suite..."
    if [ "$DRY_RUN" = false ]; then
        python -m pytest tests/ -v --cov=probneural_operator --cov-fail-under=80
        if [ $? -ne 0 ]; then
            echo_error "Tests failed. Aborting release."
            exit 1
        fi
        echo_success "All tests passed"
    else
        echo_info "[DRY RUN] Would run tests"
    fi
else
    echo_warning "Skipping tests"
fi

# Run linting and type checking
echo_info "Running code quality checks..."
if [ "$DRY_RUN" = false ]; then
    ruff check probneural_operator tests
    black --check probneural_operator tests
    mypy probneural_operator
    if [ $? -ne 0 ]; then
        echo_error "Code quality checks failed. Aborting release."
        exit 1
    fi
    echo_success "Code quality checks passed"
else
    echo_info "[DRY RUN] Would run code quality checks"
fi

# Update version in pyproject.toml
echo_info "Updating version in pyproject.toml..."
if [ "$DRY_RUN" = false ]; then
    # Use sed to update version in pyproject.toml
    sed -i.bak "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
    rm pyproject.toml.bak
    echo_success "Version updated to $NEW_VERSION"
else
    echo_info "[DRY RUN] Would update version to $NEW_VERSION"
fi

# Generate changelog entry
echo_info "Generating changelog entry..."
CHANGELOG_ENTRY="## [$NEW_VERSION] - $(date +%Y-%m-%d)

### Added
- Version $NEW_VERSION release

### Changed
- Updated dependencies and documentation

### Fixed
- Bug fixes and improvements

"

if [ "$DRY_RUN" = false ]; then
    # Prepend to CHANGELOG.md
    echo "$CHANGELOG_ENTRY$(cat CHANGELOG.md)" > CHANGELOG.md
    echo_success "Changelog updated"
else
    echo_info "[DRY RUN] Would add changelog entry:"
    echo "$CHANGELOG_ENTRY"
fi

# Build packages
if [ "$SKIP_BUILD" = false ]; then
    echo_info "Building packages..."
    if [ "$DRY_RUN" = false ]; then
        # Clean previous builds
        rm -rf build/ dist/ *.egg-info/
        
        # Build source and wheel distributions
        python -m build
        
        if [ $? -eq 0 ]; then
            echo_success "Packages built successfully"
            
            # List built packages
            echo_info "Built packages:"
            ls -la dist/
        else
            echo_error "Package build failed. Aborting release."
            exit 1
        fi
    else
        echo_info "[DRY RUN] Would build packages"
    fi
else
    echo_warning "Skipping package build"
fi

# Validate built packages
if [ "$SKIP_BUILD" = false ] && [ "$DRY_RUN" = false ]; then
    echo_info "Validating built packages..."
    python -m twine check dist/*
    if [ $? -eq 0 ]; then
        echo_success "Package validation passed"
    else
        echo_error "Package validation failed. Aborting release."
        exit 1
    fi
fi

# Commit changes
echo_info "Committing changes..."
if [ "$DRY_RUN" = false ]; then
    git add pyproject.toml CHANGELOG.md
    git commit -m "chore: release version $NEW_VERSION

- Bump version to $NEW_VERSION
- Update changelog
- Prepare for release

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
    
    echo_success "Changes committed"
else
    echo_info "[DRY RUN] Would commit version and changelog changes"
fi

# Create and push git tag
echo_info "Creating git tag..."
if [ "$DRY_RUN" = false ]; then
    git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"
    echo_success "Git tag v$NEW_VERSION created"
else
    echo_info "[DRY RUN] Would create git tag v$NEW_VERSION"
fi

# Push changes and tags
echo_info "Pushing changes to remote..."
if [ "$DRY_RUN" = false ]; then
    git push origin $BRANCH
    git push origin "v$NEW_VERSION"
    echo_success "Changes and tags pushed to remote"
else
    echo_info "[DRY RUN] Would push changes and tags to remote"
fi

# Build and tag Docker images
echo_info "Building Docker images for release..."
if [ "$DRY_RUN" = false ]; then
    # Build all target types with version tag
    for target in development production testing docs; do
        echo_info "Building Docker image: $target"
        docker build --target $target -t "probneural-operator-lab:$NEW_VERSION-$target" .
        docker tag "probneural-operator-lab:$NEW_VERSION-$target" "probneural-operator-lab:latest-$target"
    done
    echo_success "Docker images built and tagged"
else
    echo_info "[DRY RUN] Would build and tag Docker images"
fi

echo_success "Release process completed successfully!"

echo ""
echo_info "Release Summary:"
echo "• Version: $CURRENT_VERSION → $NEW_VERSION"
echo "• Git tag: v$NEW_VERSION"
echo "• Branch: $BRANCH"

if [ "$DRY_RUN" = false ]; then
    echo ""
    echo_info "Next steps:"
    echo "1. Verify the release on GitHub"
    echo "2. Upload to PyPI: twine upload dist/*"
    echo "3. Update documentation website"
    echo "4. Announce the release"
    
    # Show what files were created
    echo ""
    echo_info "Generated files:"
    if [ "$SKIP_BUILD" = false ]; then
        echo "• Package files in dist/"
        ls dist/ | sed 's/^/  /'
    fi
    
    echo ""
    echo_success "🎉 Release v$NEW_VERSION is ready!"
else
    echo ""
    echo_info "This was a dry run. No changes were made."
    echo "Run without --dry-run to perform the actual release."
fi