# GitHub Templates

This directory contains GitHub templates for workflows, issues, and pull requests that can be copied to the `.github/` directory.

## Contents

### Workflows (`workflows/`)

- `ci.yml` - Comprehensive CI/CD pipeline with testing, linting, security scans, and documentation building
- `release.yml` - Automated release process with multi-platform builds, Docker images, and PyPI publishing
- `dependency-update.yml` - Automated dependency updates with security scanning

### Issue Templates (`ISSUE_TEMPLATE/`)

- `bug_report.yml` - Structured bug report template
- `feature_request.yml` - Feature request template with detailed planning sections
- `documentation.yml` - Documentation improvement template
- `question.yml` - Question/support template

### Pull Request Template

- `pull_request_template.md` - Comprehensive PR template with quality checklists

## Installation

To use these templates, copy them to your `.github/` directory:

```bash
# Copy all templates
cp -r templates/github/.github/* .github/

# Or copy individual components
cp templates/github/workflows/* .github/workflows/
cp templates/github/ISSUE_TEMPLATE/* .github/ISSUE_TEMPLATE/
cp templates/github/pull_request_template.md .github/
```

## Customization

### Workflow Customization

1. **Update repository references** in workflows (e.g., Docker image names)
2. **Configure secrets** in GitHub repository settings:
   - `DOCKER_USERNAME` and `DOCKER_PASSWORD` for Docker Hub
   - `PYPI_API_TOKEN` for PyPI publishing
3. **Adjust test commands** to match your project structure
4. **Modify build targets** and platforms as needed

### Issue Template Customization

1. **Update labels** to match your project's label scheme
2. **Modify dropdown options** to fit your project categories
3. **Adjust required fields** based on your needs

### Pull Request Template Customization

1. **Update checklist items** to match your project requirements
2. **Modify review focus areas** for your domain
3. **Adjust testing instructions** for your project

## Features

### CI/CD Pipeline Features

- **Multi-platform testing** (Ubuntu, Windows, macOS)
- **Python version matrix** (3.9, 3.10, 3.11)
- **Comprehensive quality checks** (linting, formatting, type checking)
- **Security scanning** (Bandit, Safety)
- **Performance benchmarking**
- **Documentation building and link checking**
- **GPU testing support** (with self-hosted runners)
- **Automated releases** with semantic versioning

### Issue Template Features

- **Structured data collection** with required and optional fields
- **Dropdown selections** for categorization
- **Environment information** templates
- **Contribution willingness** tracking
- **Related work** sections for context

### Pull Request Template Features

- **Comprehensive checklists** for quality assurance
- **Security review** sections
- **Performance impact** assessment
- **Breaking change** documentation
- **Testing instruction** templates
- **Review focus** guidance

## Notes

- **GitHub App Permissions**: Some GitHub Apps may not have permission to create workflow files directly. In such cases, these templates can be manually copied.
- **Secrets Management**: Ensure all required secrets are configured before enabling workflows.
- **Runner Requirements**: GPU tests require self-hosted runners with GPU access.
- **Branch Protection**: Consider enabling branch protection rules that require these CI checks to pass.

## Best Practices

1. **Test workflows** in a fork before deploying to main repository
2. **Start with basic workflows** and add complexity gradually
3. **Monitor workflow costs** especially for matrix builds
4. **Keep templates updated** as project requirements evolve
5. **Document custom modifications** for team members