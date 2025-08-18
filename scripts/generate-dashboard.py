#!/usr/bin/env python3
"""
Repository health dashboard generator for ProbNeural Operator Lab.
Creates HTML dashboard with comprehensive project metrics and visualizations.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Optional dependencies for enhanced visualization
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - basic dashboard only")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not available - basic charts only")


class DashboardGenerator:
    """Generate comprehensive project dashboard."""
    
    def __init__(self, metrics_file: str = ".github/project-metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.metrics = self._load_metrics()
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                return json.load(f)
        return {}
    
    def _calculate_health_score(self) -> float:
        """Calculate overall repository health score."""
        if not self.metrics:
            return 0.0
        
        scores = []
        weights = []
        
        # Extract component scores from metrics
        health_config = self.metrics.get("metrics", {}).get("repository", {}).get("health_score", {})
        components = health_config.get("components", {})
        
        for component_name, component_data in components.items():
            weight = component_data.get("weight", 1)
            value = component_data.get("value", 0)
            
            # Normalize value to 0-100 scale
            if isinstance(value, (int, float)):
                normalized_value = min(100, max(0, value))
                scores.append(normalized_value)
                weights.append(weight)
        
        # Calculate weighted average
        if scores and weights:
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return 0.0
    
    def _generate_trend_data(self, metric_name: str, days: int = 30) -> List[Dict[str, Any]]:
        """Generate mock trend data for visualization."""
        # In a real implementation, this would pull historical data
        base_date = datetime.now() - timedelta(days=days)
        trend_data = []
        
        # Mock data generation - replace with real historical data
        for i in range(days):
            date = base_date + timedelta(days=i)
            # Generate realistic-looking trend data
            value = 80 + 10 * np.sin(i * 0.2) + np.random.normal(0, 2) if HAS_MATPLOTLIB else 80
            trend_data.append({
                "date": date.isoformat(),
                "value": max(0, min(100, value))
            })
        
        return trend_data
    
    def _create_health_score_chart(self) -> str:
        """Create health score visualization."""
        health_score = self._calculate_health_score()
        
        if HAS_PLOTLY:
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Repository Health Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            return fig.to_html(include_plotlyjs='cdn', div_id="health-score-chart")
        else:
            # Fallback to simple HTML representation
            color = "red" if health_score < 50 else "orange" if health_score < 80 else "green"
            return f"""
            <div class="health-score">
                <h3>Repository Health Score</h3>
                <div class="score-display" style="color: {color}; font-size: 2em; font-weight: bold;">
                    {health_score:.1f}/100
                </div>
            </div>
            """
    
    def _create_metrics_overview(self) -> str:
        """Create metrics overview section."""
        if not self.metrics:
            return "<p>No metrics data available</p>"
        
        html = """
        <div class="metrics-overview">
            <h2>Metrics Overview</h2>
            <div class="metrics-grid">
        """
        
        # Key metrics to display
        key_metrics = {
            "Test Coverage": self.metrics.get("metrics", {}).get("code_quality", {}).get("test_coverage", {}).get("overall", {}).get("value", 0),
            "Security Score": 100 - len(self.metrics.get("metrics", {}).get("security", {}).get("vulnerability_count", {}).get("critical", {}).get("count", 0)) * 20,
            "GitHub Stars": self.metrics.get("metrics", {}).get("community", {}).get("adoption_metrics", {}).get("github_stars", {}).get("current", 0),
            "Open Issues": self.metrics.get("metrics", {}).get("community", {}).get("engagement_metrics", {}).get("issues", {}).get("open", 0),
        }
        
        for metric_name, value in key_metrics.items():
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.1f}" if isinstance(value, float) else str(value)
                if "%" in metric_name.lower() or "coverage" in metric_name.lower():
                    formatted_value += "%"
                
                html += f"""
                <div class="metric-card">
                    <h4>{metric_name}</h4>
                    <div class="metric-value">{formatted_value}</div>
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _create_trend_charts(self) -> str:
        """Create trend visualizations."""
        if not HAS_PLOTLY:
            return "<p>Trend charts require Plotly. Install with: pip install plotly</p>"
        
        # Create subplot with multiple trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Test Coverage', 'Security Score', 'Build Times', 'Issue Resolution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Generate trend data for different metrics
        trends = {
            'Test Coverage': self._generate_trend_data('test_coverage'),
            'Security Score': self._generate_trend_data('security_score'),
            'Build Times': self._generate_trend_data('build_times'),
            'Issue Resolution': self._generate_trend_data('issue_resolution')
        }
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (metric_name, trend_data) in enumerate(trends.items()):
            row, col = positions[i]
            
            dates = [item['date'] for item in trend_data]
            values = [item['value'] for item in trend_data]
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, mode='lines+markers', name=metric_name),
                row=row, col=col
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="30-Day Trends")
        
        return f"""
        <div class="trends-section">
            <h2>Trend Analysis</h2>
            {fig.to_html(include_plotlyjs='cdn', div_id="trends-chart")}
        </div>
        """
    
    def _create_security_section(self) -> str:
        """Create security metrics section."""
        security_metrics = self.metrics.get("metrics", {}).get("security", {})
        
        if not security_metrics:
            return "<p>No security metrics available</p>"
        
        html = """
        <div class="security-section">
            <h2>Security Status</h2>
            <div class="security-grid">
        """
        
        # Vulnerability counts
        vuln_counts = security_metrics.get("vulnerability_count", {})
        for severity in ["critical", "high", "medium", "low"]:
            count = vuln_counts.get(severity, {}).get("count", 0)
            target = vuln_counts.get(severity, {}).get("target", 0)
            
            status = "✅" if count <= target else "⚠️" if count <= target * 2 else "❌"
            color = "green" if count <= target else "orange" if count <= target * 2 else "red"
            
            html += f"""
            <div class="security-card">
                <h4>{severity.title()} Vulnerabilities {status}</h4>
                <div class="security-value" style="color: {color};">
                    {count} / {target}
                </div>
            </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _create_quality_section(self) -> str:
        """Create code quality section."""
        quality_metrics = self.metrics.get("metrics", {}).get("code_quality", {})
        
        if not quality_metrics:
            return "<p>No code quality metrics available</p>"
        
        html = """
        <div class="quality-section">
            <h2>Code Quality</h2>
            <div class="quality-grid">
        """
        
        # Test coverage
        coverage = quality_metrics.get("test_coverage", {}).get("overall", {})
        if coverage:
            value = coverage.get("value", 0)
            target = coverage.get("target", 90)
            status = "✅" if value >= target else "⚠️" if value >= target * 0.8 else "❌"
            
            html += f"""
            <div class="quality-card">
                <h4>Test Coverage {status}</h4>
                <div class="quality-value">{value:.1f}%</div>
                <div class="quality-target">Target: {target}%</div>
            </div>
            """
        
        # Type coverage
        type_coverage = quality_metrics.get("type_coverage", {})
        if type_coverage:
            value = type_coverage.get("value", 0)
            target = type_coverage.get("target", 80)
            status = "✅" if value >= target else "⚠️" if value >= target * 0.8 else "❌"
            
            html += f"""
            <div class="quality-card">
                <h4>Type Coverage {status}</h4>
                <div class="quality-value">{value:.1f}%</div>
                <div class="quality-target">Target: {target}%</div>
            </div>
            """
        
        # Linting issues
        linting = quality_metrics.get("linting_issues", {})
        for severity in ["critical", "high", "medium"]:
            issue_data = linting.get(severity, {})
            if issue_data:
                count = issue_data.get("count", 0)
                target = issue_data.get("target", 0)
                status = "✅" if count <= target else "❌"
                
                html += f"""
                <div class="quality-card">
                    <h4>{severity.title()} Issues {status}</h4>
                    <div class="quality-value">{count}</div>
                    <div class="quality-target">Target: ≤{target}</div>
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _create_community_section(self) -> str:
        """Create community metrics section."""
        community_metrics = self.metrics.get("metrics", {}).get("community", {})
        
        if not community_metrics:
            return "<p>No community metrics available</p>"
        
        html = """
        <div class="community-section">
            <h2>Community Health</h2>
            <div class="community-grid">
        """
        
        # Adoption metrics
        adoption = community_metrics.get("adoption_metrics", {})
        for metric_name, metric_data in adoption.items():
            if isinstance(metric_data, dict) and "current" in metric_data:
                current = metric_data.get("current", 0)
                growth = metric_data.get("monthly_growth", 0)
                
                html += f"""
                <div class="community-card">
                    <h4>{metric_name.replace('_', ' ').title()}</h4>
                    <div class="community-value">{current}</div>
                    <div class="community-growth">
                        {"📈" if growth > 0 else "📉" if growth < 0 else "➡️"} {growth:+d}/month
                    </div>
                </div>
                """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _create_footer(self) -> str:
        """Create dashboard footer."""
        last_updated = self.metrics.get("last_updated", datetime.now().isoformat())
        
        return f"""
        <footer class="dashboard-footer">
            <div class="footer-content">
                <p>Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p>Metrics last updated: {last_updated}</p>
                <p>
                    <a href="https://github.com/danieleschmidt/probneural-operator-lab">View Repository</a> |
                    <a href="https://github.com/danieleschmidt/probneural-operator-lab/actions">CI/CD Status</a> |
                    <a href="https://github.com/danieleschmidt/probneural-operator-lab/security">Security</a>
                </p>
            </div>
        </footer>
        """
    
    def generate_dashboard(self, output_file: str = "dashboard.html") -> str:
        """Generate complete HTML dashboard."""
        self.logger.info("Generating repository health dashboard...")
        
        # CSS styles
        css_styles = """
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background-color: #f5f5f5; 
            }
            .dashboard-header { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 30px; 
                border-radius: 10px; 
                margin-bottom: 30px; 
                text-align: center;
            }
            .dashboard-header h1 { margin: 0; font-size: 2.5em; }
            .dashboard-header p { margin: 10px 0 0 0; opacity: 0.9; }
            
            .section { 
                background: white; 
                padding: 25px; 
                margin-bottom: 25px; 
                border-radius: 10px; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }
            .section h2 { 
                margin-top: 0; 
                color: #333; 
                border-bottom: 2px solid #eee; 
                padding-bottom: 10px; 
            }
            
            .metrics-grid, .security-grid, .quality-grid, .community-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
                margin-top: 20px; 
            }
            
            .metric-card, .security-card, .quality-card, .community-card { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px; 
                text-align: center; 
                border-left: 4px solid #667eea;
            }
            .metric-card h4, .security-card h4, .quality-card h4, .community-card h4 { 
                margin: 0 0 10px 0; 
                color: #555; 
                font-size: 0.9em;
            }
            .metric-value, .security-value, .quality-value, .community-value { 
                font-size: 2em; 
                font-weight: bold; 
                color: #333; 
                margin: 10px 0;
            }
            .quality-target, .community-growth { 
                font-size: 0.8em; 
                color: #666; 
                margin-top: 5px;
            }
            
            .health-score { 
                text-align: center; 
                padding: 30px; 
                background: white; 
                border-radius: 10px; 
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .health-score h3 { 
                margin-top: 0; 
                color: #333; 
            }
            
            .dashboard-footer { 
                margin-top: 50px; 
                text-align: center; 
                padding: 30px; 
                background: #333; 
                color: white; 
                border-radius: 10px;
            }
            .dashboard-footer a { 
                color: #667eea; 
                text-decoration: none; 
            }
            .dashboard-footer a:hover { 
                text-decoration: underline; 
            }
            
            @media (max-width: 768px) {
                .metrics-grid, .security-grid, .quality-grid, .community-grid { 
                    grid-template-columns: 1fr; 
                }
                body { padding: 10px; }
                .dashboard-header { padding: 20px; }
                .dashboard-header h1 { font-size: 2em; }
            }
        </style>
        """
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ProbNeural Operator Lab - Dashboard</title>
            {css_styles}
        </head>
        <body>
            <div class="dashboard-header">
                <h1>ProbNeural Operator Lab</h1>
                <p>Repository Health Dashboard & Metrics</p>
            </div>
            
            {self._create_health_score_chart()}
            
            <div class="section">
                {self._create_metrics_overview()}
            </div>
            
            <div class="section">
                {self._create_quality_section()}
            </div>
            
            <div class="section">
                {self._create_security_section()}
            </div>
            
            <div class="section">
                {self._create_community_section()}
            </div>
            
            {self._create_trend_charts()}
            
            {self._create_footer()}
        </body>
        </html>
        """
        
        # Write to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Dashboard generated: {output_path.absolute()}")
        return str(output_path.absolute())


def main():
    """Main entry point for dashboard generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate repository health dashboard")
    parser.add_argument("--metrics", default=".github/project-metrics.json",
                       help="Path to metrics configuration file")
    parser.add_argument("--output", default="dashboard.html",
                       help="Output HTML file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = DashboardGenerator(args.metrics)
        output_file = generator.generate_dashboard(args.output)
        
        print(f"✅ Dashboard generated successfully: {output_file}")
        print(f"   Open in browser: file://{output_file}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Error generating dashboard: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())