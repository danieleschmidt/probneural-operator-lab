#!/bin/bash

# Software Bill of Materials (SBOM) Generation Script
# Generates comprehensive SBOM for ProbNeural Operator Lab

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
OUTPUT_DIR="sbom"
FORMAT="spdx-json"
INCLUDE_CONTAINERS=true
INCLUDE_SOURCE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -f|--format)
            FORMAT="$2"
            shift 2
            ;;
        --no-containers)
            INCLUDE_CONTAINERS=false
            shift
            ;;
        --no-source)
            INCLUDE_SOURCE=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -o, --output DIR     Output directory (default: sbom)"
            echo "  -f, --format FORMAT  SBOM format: spdx-json, cyclonedx-json (default: spdx-json)"
            echo "  --no-containers      Skip container SBOM generation"
            echo "  --no-source          Skip source code SBOM generation"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo_error "Unknown option $1"
            exit 1
            ;;
    esac
done

echo_info "Generating Software Bill of Materials (SBOM)"
echo_info "Output directory: $OUTPUT_DIR"
echo_info "Format: $FORMAT"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get project information
PROJECT_NAME="probneural-operator-lab"
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "dev")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

echo_info "Project: $PROJECT_NAME"
echo_info "Version: $VERSION"

# Generate Python dependencies SBOM
echo_info "Generating Python dependencies SBOM..."

if [ "$INCLUDE_SOURCE" = true ]; then
    # Generate requirements.txt from current environment
    echo_info "Collecting Python dependencies..."
    pip freeze > "$OUTPUT_DIR/requirements.txt"
    
    # Use pip-licenses to get license information
    if command -v pip-licenses >/dev/null 2>&1; then
        pip-licenses --format=json --output-file="$OUTPUT_DIR/python-licenses.json"
    else
        echo_warning "pip-licenses not installed, skipping license information"
    fi
    
    # Create SPDX document for Python dependencies
    cat > "$OUTPUT_DIR/python-dependencies-$FORMAT" << EOF
{
  "spdxVersion": "SPDX-2.3",
  "dataLicense": "CC0-1.0",
  "SPDXID": "SPDXRef-DOCUMENT",
  "name": "$PROJECT_NAME-python-dependencies",
  "documentNamespace": "https://github.com/danieleschmidt/probneural-operator-lab/sbom/python-$VERSION",
  "creator": ["Tool: generate-sbom.sh"],
  "created": "$TIMESTAMP",
  "packageVerificationCode": {
    "packageVerificationCodeValue": "$(sha1sum requirements.txt | cut -d' ' -f1)"
  },
  "packages": [
EOF
    
    # Parse pip freeze output and add to SBOM
    echo_info "Processing dependency information..."
    FIRST=true
    while IFS='==' read -r package version || [ -n "$package" ]; do
        if [ -n "$package" ] && [ -n "$version" ]; then
            if [ "$FIRST" = false ]; then
                echo "," >> "$OUTPUT_DIR/python-dependencies-$FORMAT"
            fi
            FIRST=false
            
            cat >> "$OUTPUT_DIR/python-dependencies-$FORMAT" << EOF
    {
      "SPDXID": "SPDXRef-Package-$(echo $package | tr '[:lower:]' '[:upper:]' | tr '-' '_')",
      "name": "$package",
      "downloadLocation": "https://pypi.org/project/$package/$version/",
      "filesAnalyzed": false,
      "versionInfo": "$version",
      "supplier": "NOASSERTION",
      "copyrightText": "NOASSERTION"
    }
EOF
        fi
    done < <(grep -E '^[^#].*==' "$OUTPUT_DIR/requirements.txt")
    
    cat >> "$OUTPUT_DIR/python-dependencies-$FORMAT" << EOF
  ]
}
EOF
    
    echo_success "Python dependencies SBOM generated"
fi

# Generate container SBOM if requested
if [ "$INCLUDE_CONTAINERS" = true ]; then
    echo_info "Generating container SBOM..."
    
    # Check if syft is available
    if command -v syft >/dev/null 2>&1; then
        # Generate SBOM for each Docker target
        for target in development production testing docs; do
            IMAGE_NAME="probneural-operator-lab:latest-$target"
            
            # Check if image exists
            if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
                echo_info "Generating SBOM for $IMAGE_NAME..."
                syft packages "$IMAGE_NAME" -o "$FORMAT" > "$OUTPUT_DIR/container-$target-$FORMAT.json"
                echo_success "Container SBOM generated for $target"
            else
                echo_warning "Container image $IMAGE_NAME not found, skipping"
            fi
        done
    else
        echo_warning "Syft not installed, skipping container SBOM generation"
        echo_info "Install syft: curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin"
    fi
fi

# Generate source code SBOM
if [ "$INCLUDE_SOURCE" = true ]; then
    echo_info "Generating source code SBOM..."
    
    # Create source code inventory
    echo_info "Analyzing source code structure..."
    
    # Count lines of code by language
    if command -v cloc >/dev/null 2>&1; then
        cloc . --json --out="$OUTPUT_DIR/source-analysis.json"
        echo_success "Source code analysis completed"
    else
        echo_warning "cloc not installed, skipping source code analysis"
    fi
    
    # Generate file manifest
    echo_info "Creating file manifest..."
    find . -type f -not -path './.git/*' -not -path './venv/*' -not -path './__pycache__/*' | \
        sort > "$OUTPUT_DIR/file-manifest.txt"
    
    # Calculate checksums for important files
    echo_info "Calculating checksums..."
    {
        echo "# File checksums for $PROJECT_NAME v$VERSION"
        echo "# Generated: $TIMESTAMP"
        echo ""
        
        # Calculate SHA256 for Python source files
        find . -name "*.py" -not -path './venv/*' -not -path './.git/*' | \
            xargs sha256sum 2>/dev/null || true
    } > "$OUTPUT_DIR/checksums.txt"
    
    echo_success "Source code SBOM generated"
fi

# Generate vulnerability report if tools are available
echo_info "Generating vulnerability assessment..."

if command -v safety >/dev/null 2>&1; then
    echo_info "Running Python dependency vulnerability scan..."
    safety check --json --output "$OUTPUT_DIR/vulnerability-report.json" || true
    echo_success "Vulnerability scan completed"
else
    echo_warning "Safety not installed, skipping vulnerability scan"
fi

# Create comprehensive SBOM summary
echo_info "Creating SBOM summary..."

cat > "$OUTPUT_DIR/sbom-summary.json" << EOF
{
  "sbom": {
    "project": "$PROJECT_NAME",
    "version": "$VERSION",
    "generated": "$TIMESTAMP",
    "format": "$FORMAT",
    "tools": [
      "generate-sbom.sh"
    ]
  },
  "components": {
    "source_code": $INCLUDE_SOURCE,
    "python_dependencies": $INCLUDE_SOURCE,
    "containers": $INCLUDE_CONTAINERS
  },
  "files": [
EOF

# List all generated files
FIRST=true
for file in "$OUTPUT_DIR"/*; do
    if [ -f "$file" ] && [ "$(basename "$file")" != "sbom-summary.json" ]; then
        if [ "$FIRST" = false ]; then
            echo "," >> "$OUTPUT_DIR/sbom-summary.json"
        fi
        FIRST=false
        
        filename=$(basename "$file")
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "unknown")
        checksum=$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1 || echo "unknown")
        
        cat >> "$OUTPUT_DIR/sbom-summary.json" << EOF
    {
      "name": "$filename",
      "size": $size,
      "sha256": "$checksum"
    }
EOF
    fi
done

cat >> "$OUTPUT_DIR/sbom-summary.json" << EOF
  ]
}
EOF

# Generate attestation
echo_info "Creating attestation..."

cat > "$OUTPUT_DIR/attestation.txt" << EOF
Software Bill of Materials (SBOM) Attestation

Project: $PROJECT_NAME
Version: $VERSION
Generated: $TIMESTAMP
Generator: generate-sbom.sh (ProbNeural Operator Lab)

This SBOM was generated using automated tools and represents the software
components and dependencies present in the $PROJECT_NAME project at the
time of generation.

Components Included:
- Source code analysis: $INCLUDE_SOURCE
- Python dependencies: $INCLUDE_SOURCE  
- Container images: $INCLUDE_CONTAINERS

Verification:
The checksums in checksums.txt can be used to verify the integrity of
the source code files included in this SBOM.

For questions about this SBOM, contact: security@example.com

Generated by: $(whoami)@$(hostname)
Git commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
EOF

echo_success "SBOM generation completed!"

# Display summary
echo ""
echo_info "SBOM Summary:"
echo "• Output directory: $OUTPUT_DIR"
echo "• Format: $FORMAT"
echo "• Files generated:"

for file in "$OUTPUT_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "unknown")
        echo "  - $filename ($size bytes)"
    fi
done

echo ""
echo_info "Next steps:"
echo "1. Review the generated SBOM files in $OUTPUT_DIR/"
echo "2. Store the SBOM in a secure, version-controlled location"
echo "3. Share with security team and compliance officers"
echo "4. Include SBOM in release artifacts"
echo "5. Update SBOM regularly as dependencies change"

echo ""
echo_success "🔒 SBOM generation complete!"