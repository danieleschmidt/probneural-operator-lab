#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Value Discovery Engine
Continuously discovers, scores, and prioritizes repository improvements.
"""

import json
import subprocess
import re
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

class TerragnonValueDiscovery:
    """Autonomous value discovery and prioritization system."""
    
    def __init__(self, config_path: str = ".terragon/config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.discovered_items = []
        self.scoring_history = []
        
    def load_config(self) -> dict:
        """Load Terragon configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Terragon config not found at {self.config_path}")
        
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def run_comprehensive_discovery(self) -> Dict[str, any]:
        """Run complete value discovery cycle."""
        print("🔍 Starting Terragon Autonomous Value Discovery...")
        
        discovery_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'sources': {},
            'discovered_items': [],
            'metrics': {}
        }
        
        # Run all discovery sources
        discovery_sources = [
            ('git_history', self.discover_from_git_history),
            ('static_analysis', self.discover_from_static_analysis),
            ('dependency_audit', self.discover_from_dependencies),
            ('security_scan', self.discover_from_security),
            ('performance_analysis', self.discover_from_performance),
            ('documentation_gaps', self.discover_documentation_gaps)
        ]
        
        for source_name, discovery_func in discovery_sources:
            print(f"  📊 Running {source_name} discovery...")
            try:
                source_results = discovery_func()
                discovery_results['sources'][source_name] = source_results
                print(f"    ✅ Found {len(source_results.get('items', []))} items")
            except Exception as e:
                print(f"    ❌ Error in {source_name}: {e}")
                discovery_results['sources'][source_name] = {'error': str(e)}
        
        # Aggregate and score all discovered items
        all_items = []
        for source_data in discovery_results['sources'].values():
            if isinstance(source_data, dict) and 'items' in source_data:
                all_items.extend(source_data['items'])
        
        # Score and prioritize items
        scored_items = self.score_and_prioritize_items(all_items)
        discovery_results['discovered_items'] = scored_items
        
        # Calculate metrics
        discovery_results['metrics'] = self.calculate_discovery_metrics(discovery_results)
        
        # Save results
        self.save_discovery_results(discovery_results)
        
        # Update backlog
        self.update_backlog(scored_items)
        
        print(f"🎯 Discovery complete! Found {len(scored_items)} prioritized items")
        return discovery_results
    
    def discover_from_git_history(self) -> Dict[str, any]:
        """Discover tasks from Git history analysis."""
        items = []
        
        try:
            # Search for TODO/FIXME/HACK patterns
            result = subprocess.run([
                'git', 'grep', '-n', '-i', '-E', 
                '(TODO|FIXME|HACK|XXX|NOTE|BUG):', 
                '--', '*.py', '*.md', '*.yaml', '*.yml'
            ], capture_output=True, text=True)
            
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    match = re.match(r'([^:]+):(\d+):(.*)', line)
                    if match:
                        file_path, line_num, content = match.groups()
                        
                        # Extract TODO type and description
                        todo_match = re.search(r'(TODO|FIXME|HACK|XXX|NOTE|BUG):\s*(.*)', content, re.IGNORECASE)
                        if todo_match:
                            todo_type, description = todo_match.groups()
                            
                            items.append({
                                'id': f'GIT-{len(items)+1:03d}',
                                'title': f'Address {todo_type.lower()}: {description[:50]}...',
                                'description': description.strip(),
                                'source': 'git_history',
                                'type': 'technical_debt',
                                'priority': self.classify_todo_priority(todo_type),
                                'location': f'{file_path}:{line_num}',
                                'estimated_effort': self.estimate_todo_effort(description),
                                'discovery_timestamp': datetime.utcnow().isoformat()
                            })
            
            # Analyze commit messages for patterns
            commit_result = subprocess.run([
                'git', 'log', '--oneline', '--since=30 days ago', 
                '--grep=fix', '--grep=hack', '--grep=temp', '--grep=quick'
            ], capture_output=True, text=True)
            
            if commit_result.stdout:
                quick_fixes = len(commit_result.stdout.strip().split('\n'))
                if quick_fixes > 5:
                    items.append({
                        'id': f'GIT-PATTERN-001',
                        'title': f'Refactor areas with frequent quick fixes ({quick_fixes} recent fixes)',
                        'description': 'High frequency of quick fixes indicates technical debt areas needing refactoring',
                        'source': 'git_history',
                        'type': 'technical_debt',
                        'priority': 'medium',
                        'estimated_effort': quick_fixes * 0.5,  # 30 min per fix to properly address
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
        
        except subprocess.CalledProcessError:
            pass  # Git commands may fail in some environments
        
        return {
            'items': items,
            'stats': {
                'total_todos': len([i for i in items if 'TODO' in i.get('title', '')]),
                'total_fixmes': len([i for i in items if 'FIXME' in i.get('title', '')]),
                'total_hacks': len([i for i in items if 'HACK' in i.get('title', '')])
            }
        }
    
    def discover_from_static_analysis(self) -> Dict[str, any]:
        """Discover issues from static code analysis."""
        items = []
        
        # Run ruff for code quality issues  
        try:
            result = subprocess.run([
                'ruff', 'check', 'probneural_operator', '--format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                
                # Group issues by type
                issue_groups = {}
                for issue in ruff_issues:
                    rule_code = issue.get('code', 'unknown')
                    if rule_code not in issue_groups:
                        issue_groups[rule_code] = []
                    issue_groups[rule_code].append(issue)
                
                # Create items for issue groups with multiple occurrences
                for rule_code, issues in issue_groups.items():
                    if len(issues) >= 3:  # Only create items for patterns
                        items.append({
                            'id': f'STATIC-{rule_code}',
                            'title': f'Fix {len(issues)} instances of {rule_code}',
                            'description': f'Code quality improvement: {issues[0].get("message", "")}',
                            'source': 'static_analysis',
                            'type': 'code_quality',
                            'priority': self.classify_ruff_priority(rule_code),
                            'locations': [f'{i.get("filename")}:{i.get("location", {}).get("row")}' for i in issues[:5]],
                            'estimated_effort': len(issues) * 0.1,  # 6 minutes per fix
                            'discovery_timestamp': datetime.utcnow().isoformat()
                        })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        
        # Run mypy for type issues
        try:
            result = subprocess.run([
                'mypy', 'probneural_operator', '--json-report', '/tmp/mypy-report'
            ], capture_output=True, text=True)
            
            mypy_report_path = Path('/tmp/mypy-report/index.txt')
            if mypy_report_path.exists():
                with open(mypy_report_path) as f:
                    mypy_output = f.read()
                    
                # Count type errors
                type_errors = mypy_output.count('error:')
                if type_errors > 5:
                    items.append({
                        'id': 'STATIC-TYPES-001',
                        'title': f'Improve type annotations ({type_errors} type issues)',
                        'description': 'Add missing type annotations and fix type errors',
                        'source': 'static_analysis',
                        'type': 'code_quality',
                        'priority': 'medium',
                        'estimated_effort': type_errors * 0.15,  # 9 minutes per type fix
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
        
        except subprocess.CalledProcessError:
            pass
        
        return {
            'items': items,
            'stats': {
                'code_quality_issues': len([i for i in items if i['type'] == 'code_quality']),
                'type_issues': len([i for i in items if 'type' in i.get('title', '').lower()])
            }
        }
    
    def discover_from_dependencies(self) -> Dict[str, any]:
        """Discover dependency-related improvements."""
        items = []
        
        # Check for vulnerability
        try:
            result = subprocess.run([
                'pip-audit', '--format', 'json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                audit_data = json.loads(result.stdout)
                vulnerabilities = audit_data.get('vulnerabilities', [])
                
                for vuln in vulnerabilities:
                    severity = vuln.get('vulnerability', {}).get('severity', 'unknown')
                    package = vuln.get('package')
                    
                    items.append({
                        'id': f'DEPS-VULN-{package}',
                        'title': f'Update {package} to fix {severity} vulnerability',
                        'description': vuln.get('vulnerability', {}).get('description', ''),
                        'source': 'dependency_audit',
                        'type': 'security',
                        'priority': 'critical' if severity in ['high', 'critical'] else 'medium',
                        'estimated_effort': 0.5,  # 30 minutes to update and test
                        'cve_id': vuln.get('vulnerability', {}).get('id'),
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        # Check for outdated dependencies
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                
                if len(outdated) > 10:  # Many outdated packages
                    items.append({
                        'id': 'DEPS-OUTDATED-001',
                        'title': f'Update {len(outdated)} outdated dependencies',
                        'description': 'Bulk update of non-critical dependency versions',
                        'source': 'dependency_audit',
                        'type': 'maintenance',
                        'priority': 'low',
                        'estimated_effort': len(outdated) * 0.1,  # 6 minutes per package
                        'package_count': len(outdated),
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        
        return {
            'items': items,
            'stats': {
                'security_vulnerabilities': len([i for i in items if i['type'] == 'security']),
                'outdated_packages': len([i for i in items if 'outdated' in i.get('title', '').lower()])
            }
        }
    
    def discover_from_security(self) -> Dict[str, any]:
        """Discover security improvements."""
        items = []
        
        # Run bandit security analysis
        try:
            result = subprocess.run([
                'bandit', '-r', 'probneural_operator', '-f', 'json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                results = bandit_data.get('results', [])
                
                # Group by severity
                high_severity = [r for r in results if r.get('issue_severity') == 'HIGH']
                medium_severity = [r for r in results if r.get('issue_severity') == 'MEDIUM']
                
                if high_severity:
                    items.append({
                        'id': 'SEC-BANDIT-HIGH',
                        'title': f'Fix {len(high_severity)} high-severity security issues',
                        'description': 'Address high-severity security vulnerabilities in code',
                        'source': 'security_scan',
                        'type': 'security',
                        'priority': 'critical',
                        'estimated_effort': len(high_severity) * 0.5,  # 30 minutes per issue
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
                
                if len(medium_severity) >= 3:
                    items.append({
                        'id': 'SEC-BANDIT-MEDIUM',
                        'title': f'Fix {len(medium_severity)} medium-severity security issues',
                        'description': 'Address medium-severity security vulnerabilities',
                        'source': 'security_scan',
                        'type': 'security',
                        'priority': 'medium',
                        'estimated_effort': len(medium_severity) * 0.25,  # 15 minutes per issue
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
        
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
            pass
        
        return {
            'items': items,
            'stats': {
                'high_severity_issues': len([i for i in items if 'high-severity' in i.get('title', '')]),
                'medium_severity_issues': len([i for i in items if 'medium-severity' in i.get('title', '')])
            }
        }
    
    def discover_from_performance(self) -> Dict[str, any]:
        """Discover performance improvement opportunities."""
        items = []
        
        # Look for potential performance issues in code patterns
        try:
            # Search for potential performance anti-patterns
            patterns_to_check = [
                (r'\.append\(.*\) for .* in', 'Use list comprehension instead of append in loop'),
                (r'open\([^)]+\)(?!\s+as)', 'Use context manager (with statement) for file operations'),
                (r'\.+\*\*.*for.*in', 'Consider using numpy operations for mathematical computations')
            ]
            
            for pattern, suggestion in patterns_to_check:
                result = subprocess.run([
                    'grep', '-r', '-n', pattern, 'probneural_operator/', '--include=*.py'
                ], capture_output=True, text=True)
                
                if result.stdout:
                    matches = result.stdout.strip().split('\n')
                    if len(matches) >= 3:  # Multiple occurrences
                        items.append({
                            'id': f'PERF-PATTERN-{len(items)+1:03d}',
                            'title': f'Optimize {len(matches)} performance patterns',
                            'description': suggestion,
                            'source': 'performance_analysis',
                            'type': 'performance',
                            'priority': 'low',
                            'estimated_effort': len(matches) * 0.2,  # 12 minutes per fix
                            'pattern_count': len(matches),
                            'discovery_timestamp': datetime.utcnow().isoformat()
                        })
        
        except subprocess.CalledProcessError:
            pass
        
        return {
            'items': items,
            'stats': {
                'performance_patterns': len(items)
            }
        }
    
    def discover_documentation_gaps(self) -> Dict[str, any]:
        """Discover documentation improvements."""
        items = []
        
        # Check for missing docstrings
        try:
            result = subprocess.run([
                'python', '-c', 
                '''
import ast
import os
missing_docstrings = 0
for root, dirs, files in os.walk("probneural_operator"):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            try:
                with open(filepath) as f:
                    tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        if not ast.get_docstring(node):
                            missing_docstrings += 1
            except:
                pass
print(missing_docstrings)
                '''
            ], capture_output=True, text=True)
            
            if result.stdout.strip().isdigit():
                missing_count = int(result.stdout.strip())
                if missing_count > 10:
                    items.append({
                        'id': 'DOCS-DOCSTRINGS-001',
                        'title': f'Add docstrings to {missing_count} functions/classes',
                        'description': 'Improve code documentation with missing docstrings',
                        'source': 'documentation_gaps',
                        'type': 'documentation',
                        'priority': 'low',
                        'estimated_effort': missing_count * 0.15,  # 9 minutes per docstring
                        'discovery_timestamp': datetime.utcnow().isoformat()
                    })
        
        except subprocess.CalledProcessError:
            pass
        
        # Check for outdated documentation
        readme_path = Path('README.md')
        if readme_path.exists():
            readme_age = datetime.now() - datetime.fromtimestamp(readme_path.stat().st_mtime)
            if readme_age.days > 90:  # README older than 3 months
                items.append({
                    'id': 'DOCS-README-001',
                    'title': 'Update README.md (outdated by 3+ months)',
                    'description': 'Review and update repository documentation',
                    'source': 'documentation_gaps',
                    'type': 'documentation',
                    'priority': 'low',
                    'estimated_effort': 1.0,  # 1 hour
                    'days_outdated': readme_age.days,
                    'discovery_timestamp': datetime.utcnow().isoformat()
                })
        
        return {
            'items': items,
            'stats': {
                'documentation_gaps': len(items)
            }
        }
    
    def score_and_prioritize_items(self, items: List[dict]) -> List[dict]:
        """Score and prioritize discovered items using WSJF + ICE + Technical Debt."""
        scored_items = []
        
        for item in items:
            # Calculate WSJF components
            user_business_value = self.calculate_business_value(item)
            time_criticality = self.calculate_time_criticality(item)  
            risk_reduction = self.calculate_risk_reduction(item)
            opportunity_enablement = self.calculate_opportunity_enablement(item)
            
            cost_of_delay = (user_business_value + time_criticality + 
                           risk_reduction + opportunity_enablement)
            job_size = item.get('estimated_effort', 1.0)
            wsjf_score = cost_of_delay / job_size if job_size > 0 else 0
            
            # Calculate ICE components
            impact = self.calculate_impact(item)
            confidence = self.calculate_confidence(item)
            ease = self.calculate_ease(item)
            ice_score = impact * confidence * ease
            
            # Calculate Technical Debt score
            tech_debt_score = self.calculate_tech_debt_score(item)
            
            # Get weights for repository maturity level
            maturity_level = self.config['metadata']['maturity_level']
            weights = self.config['scoring']['weights'][maturity_level]
            
            # Calculate composite score
            composite_score = (
                weights['wsjf'] * self.normalize_score(wsjf_score, 'wsjf') +
                weights['ice'] * self.normalize_score(ice_score, 'ice') +
                weights['technicalDebt'] * self.normalize_score(tech_debt_score, 'debt') +
                weights['security'] * (2.0 if item['type'] == 'security' else 1.0)
            )
            
            # Apply security and compliance boosts
            if item['type'] == 'security':
                composite_score *= self.config['scoring']['thresholds']['securityBoost']
            
            # Create scored item
            scored_item = {
                **item,
                'scores': {
                    'wsjf': round(wsjf_score, 1),
                    'ice': round(ice_score, 1),
                    'technical_debt': round(tech_debt_score, 1),
                    'composite': round(composite_score, 1)
                },
                'scoring_components': {
                    'user_business_value': user_business_value,
                    'time_criticality': time_criticality,
                    'risk_reduction': risk_reduction,
                    'opportunity_enablement': opportunity_enablement,
                    'impact': impact,
                    'confidence': confidence,
                    'ease': ease
                }
            }
            
            scored_items.append(scored_item)
        
        # Sort by composite score
        scored_items.sort(key=lambda x: x['scores']['composite'], reverse=True)
        
        return scored_items
    
    def calculate_business_value(self, item: dict) -> float:
        """Calculate business value score (1-10)."""
        value_map = {
            'security': 9,      # High business value
            'infrastructure': 7, # Medium-high value  
            'performance': 6,    # Medium value
            'technical_debt': 5, # Medium value
            'code_quality': 4,   # Medium-low value
            'documentation': 3,  # Low-medium value
            'maintenance': 2     # Low value
        }
        return value_map.get(item['type'], 3)
    
    def calculate_time_criticality(self, item: dict) -> float:
        """Calculate time criticality score (1-10)."""
        if item['type'] == 'security':
            return 9  # Security issues are time-critical
        elif item.get('priority') == 'critical':
            return 8
        elif item.get('priority') == 'high':
            return 6
        elif item.get('priority') == 'medium':
            return 4
        else:
            return 2
    
    def calculate_risk_reduction(self, item: dict) -> float:
        """Calculate risk reduction score (1-10)."""
        if item['type'] == 'security':
            return 8
        elif item['type'] == 'infrastructure':
            return 6
        elif item['type'] == 'technical_debt':
            return 5
        else:
            return 3
    
    def calculate_opportunity_enablement(self, item: dict) -> float:
        """Calculate opportunity enablement score (1-10)."""
        if item['type'] == 'infrastructure':
            return 7  # Infrastructure enables future work
        elif item['type'] == 'performance':
            return 5
        elif item['type'] == 'code_quality':
            return 4
        else:
            return 2
    
    def calculate_impact(self, item: dict) -> float:
        """Calculate ICE impact score (1-10)."""
        return self.calculate_business_value(item)  # Same as business value for now
    
    def calculate_confidence(self, item: dict) -> float:
        """Calculate ICE confidence score (1-10)."""
        # Base confidence on source reliability and item complexity
        source_confidence = {
            'git_history': 8,        # High confidence - concrete evidence
            'static_analysis': 9,    # Very high confidence - tool-based
            'dependency_audit': 10,  # Highest confidence - automated scan
            'security_scan': 9,      # Very high confidence - tool-based  
            'performance_analysis': 6, # Medium confidence - pattern-based
            'documentation_gaps': 7   # High confidence - measurable
        }
        
        base_confidence = source_confidence.get(item['source'], 5)
        
        # Adjust based on effort estimate (higher effort = lower confidence)
        effort = item.get('estimated_effort', 1)
        if effort > 4:
            base_confidence -= 2
        elif effort > 2:
            base_confidence -= 1
            
        return max(1, min(10, base_confidence))
    
    def calculate_ease(self, item: dict) -> float:
        """Calculate ICE ease score (1-10)."""
        # Inverse of effort estimate
        effort = item.get('estimated_effort', 1)
        if effort <= 0.5:
            return 10
        elif effort <= 1:
            return 8
        elif effort <= 2:
            return 6
        elif effort <= 4:
            return 4
        else:
            return 2
    
    def calculate_tech_debt_score(self, item: dict) -> float:
        """Calculate technical debt reduction score."""
        if item['type'] == 'technical_debt':
            return 8
        elif item['type'] == 'code_quality':
            return 6
        elif item['type'] == 'performance':
            return 4
        else:
            return 1
    
    def normalize_score(self, score: float, score_type: str) -> float:
        """Normalize scores to 0-100 range."""
        if score_type == 'wsjf':
            return min(100, score * 10)  # WSJF typically 0-10
        elif score_type == 'ice':
            return min(100, score)  # ICE can be 1-1000, normalize  
        elif score_type == 'debt':
            return min(100, score * 10)  # Debt score 0-10
        else:
            return score
    
    def classify_todo_priority(self, todo_type: str) -> str:
        """Classify TODO priority based on type."""
        priority_map = {
            'FIXME': 'high',
            'BUG': 'high', 
            'HACK': 'medium',
            'TODO': 'medium',
            'XXX': 'medium',
            'NOTE': 'low'
        }
        return priority_map.get(todo_type.upper(), 'low')
    
    def classify_ruff_priority(self, rule_code: str) -> str:
        """Classify Ruff rule priority."""
        high_priority_rules = ['F', 'E9', 'W6']  # Syntax errors, serious issues
        medium_priority_rules = ['E', 'W', 'B']  # Style and bug-prone patterns
        
        rule_prefix = rule_code[0] if rule_code else ''
        
        if any(rule_code.startswith(prefix) for prefix in high_priority_rules):
            return 'high'
        elif any(rule_code.startswith(prefix) for prefix in medium_priority_rules):
            return 'medium'
        else:
            return 'low'
    
    def estimate_todo_effort(self, description: str) -> float:
        """Estimate effort for TODO items based on description."""
        description_lower = description.lower()
        
        # Simple heuristics based on keywords
        if any(word in description_lower for word in ['refactor', 'rewrite', 'redesign']):
            return 4.0  # 4 hours for major changes
        elif any(word in description_lower for word in ['fix', 'update', 'change']):
            return 1.0  # 1 hour for fixes
        elif any(word in description_lower for word in ['add', 'implement']):
            return 2.0  # 2 hours for new features
        else:
            return 0.5  # 30 minutes for simple tasks
    
    def calculate_discovery_metrics(self, results: dict) -> dict:
        """Calculate discovery performance metrics."""
        total_items = len(results['discovered_items'])
        
        by_type = {}
        by_priority = {}
        by_source = {}
        
        for item in results['discovered_items']:
            # Count by type
            item_type = item['type']
            by_type[item_type] = by_type.get(item_type, 0) + 1
            
            # Count by priority
            priority = item.get('priority', 'unknown')
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
            # Count by source
            source = item['source']
            by_source[source] = by_source.get(source, 0) + 1
        
        return {
            'total_items_discovered': total_items,
            'items_by_type': by_type,
            'items_by_priority': by_priority,
            'items_by_source': by_source,
            'discovery_efficiency': {
                source: (count / total_items) if total_items > 0 else 0
                for source, count in by_source.items()
            }
        }
    
    def save_discovery_results(self, results: dict):
        """Save discovery results to file."""
        output_file = Path('.terragon/discovery-results.json')
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📊 Discovery results saved to: {output_file}")
    
    def update_backlog(self, scored_items: List[dict]):
        """Update the BACKLOG.md file with new discoveries."""
        backlog_path = Path('BACKLOG.md')
        
        # Generate backlog content
        now = datetime.utcnow().isoformat()
        next_run = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        
        # Get top 10 items
        top_items = scored_items[:10]
        ready_items = [item for item in top_items if item['scores']['composite'] > 50]
        
        # Calculate metrics
        total_effort = sum(item.get('estimated_effort', 0) for item in ready_items)
        security_items = len([item for item in scored_items if item['type'] == 'security'])
        
        backlog_content = f"""# 📊 Autonomous Value Backlog

**Repository**: {self.config['metadata']['repository_name']}  
**Maturity Level**: {self.config['metadata']['maturity_level'].title()} ({self.config['maturity_assessment']['overall_score']}% SDLC maturity)  
**Last Updated**: {now}  
**Next Discovery Run**: {next_run}  

## 🎯 Discovery Summary

- **Total Items Discovered**: {len(scored_items)}
- **High-Value Items (Score > 50)**: {len(ready_items)}
- **Security Items**: {security_items}
- **Ready for Execution**: {len([i for i in ready_items if i['scores']['composite'] > 70])}

## 🚀 Top Priority Items (Ready for Execution)

"""
        
        for i, item in enumerate(ready_items[:5], 1):
            effort_hours = item.get('estimated_effort', 0)
            backlog_content += f"""### {i}. **[{item['id']}] {item['title']}**
- **Composite Score**: {item['scores']['composite']}
- **WSJF**: {item['scores']['wsjf']} | **ICE**: {item['scores']['ice']} | **Tech Debt**: {item['scores']['technical_debt']}
- **Estimated Effort**: {effort_hours} hours
- **Type**: {item['type'].title()}
- **Priority**: {item.get('priority', 'medium').title()}
- **Source**: {item['source'].replace('_', ' ').title()}

"""
        
        backlog_content += f"""
## 📋 Complete Backlog (Top 10)

| Rank | ID | Title | Score | Type | Priority | Hours |
|------|-----|--------|---------|------|----------|--------|
"""
        
        for i, item in enumerate(top_items, 1):
            title_short = item['title'][:50] + '...' if len(item['title']) > 50 else item['title']
            backlog_content += f"| {i} | {item['id']} | {title_short} | {item['scores']['composite']} | {item['type'].title()} | {item.get('priority', 'medium').title()} | {item.get('estimated_effort', 0)} |\n"
        
        backlog_content += f"""
## 📈 Value Metrics

- **Total Estimated Effort**: {total_effort:.1f} hours
- **Average Item Score**: {sum(item['scores']['composite'] for item in top_items) / len(top_items):.1f}
- **Security Items**: {security_items}
- **Technical Debt Items**: {len([i for i in scored_items if i['type'] == 'technical_debt'])}

## 🔄 Next Discovery Schedule

- **Next Comprehensive Scan**: {next_run}
- **Security Scan**: Every hour
- **Dependency Audit**: Daily
- **Performance Check**: Daily

---

*🤖 This backlog is automatically maintained by Terragon Autonomous SDLC system. Last discovery: {now}*
"""
        
        with open(backlog_path, 'w') as f:
            f.write(backlog_content)
        
        print(f"📋 Backlog updated: {backlog_path}")

def main():
    """Main entry point for discovery engine."""
    discovery = TerragnonValueDiscovery()
    results = discovery.run_comprehensive_discovery()
    
    print(f"\n🎯 Discovery Complete!")
    print(f"📊 Found {len(results['discovered_items'])} items")
    print(f"🏆 Top item score: {results['discovered_items'][0]['scores']['composite'] if results['discovered_items'] else 'N/A'}")
    print(f"⏱️  Next discovery in 1 hour")

if __name__ == '__main__':
    main()