"""
Report generation for gait analysis results.

This module provides classes for generating HTML and text reports summarizing
gait analysis findings with embedded visualizations and statistical summaries.
"""

import json
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ReportGenerator:
    """
    Class for generating analysis reports in various formats.

    This class creates comprehensive reports with metrics summaries,
    visualizations, and interpretations of gait analysis results.
    """

    def __init__(
        self,
        project_name: str = "Running Gait Analysis",
        author: Optional[str] = None,
    ):
        """
        Initialize the ReportGenerator.

        Args:
            project_name: Name of the analysis project
            author: Optional author name
        """
        self.project_name = project_name
        self.author = author
        self.created_at = datetime.now()

        # HTML template styles
        self.css_style = """
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .header .subtitle {
                margin-top: 10px;
                font-size: 1.2em;
                opacity: 0.9;
            }
            .section {
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .section h2 {
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .section h3 {
                color: #764ba2;
                margin-top: 20px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .metric-card {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .metric-card .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }
            .metric-card .metric-label {
                font-size: 0.9em;
                color: #666;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .metric-card .metric-unit {
                font-size: 0.8em;
                color: #888;
            }
            .plot-container {
                margin: 20px 0;
                text-align: center;
            }
            .plot-container img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            table th {
                background-color: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: 600;
            }
            table td {
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }
            table tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            table tr:hover {
                background-color: #f0f0f0;
            }
            .alert {
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }
            .alert-info {
                background-color: #d1ecf1;
                border-left: 4px solid #0c5460;
                color: #0c5460;
            }
            .alert-warning {
                background-color: #fff3cd;
                border-left: 4px solid #856404;
                color: #856404;
            }
            .alert-success {
                background-color: #d4edda;
                border-left: 4px solid #155724;
                color: #155724;
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 0.9em;
                margin-top: 40px;
            }
            .comparison-table {
                margin: 20px 0;
            }
            .side-by-side {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }
            @media print {
                body {
                    background-color: white;
                }
                .section {
                    box-shadow: none;
                    page-break-inside: avoid;
                }
            }
        </style>
        """

    def generate_html(
        self,
        metrics: Dict[str, Any],
        figures: Optional[Dict[str, plt.Figure]] = None,
        summary_text: Optional[str] = None,
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate an HTML report with metrics and visualizations.

        Args:
            metrics: Dictionary of computed metrics
            figures: Optional dictionary of matplotlib figures to include
            summary_text: Optional summary text
            output_path: Optional path to save the HTML file

        Returns:
            HTML string
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='UTF-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
            f"<title>{self.project_name} - Report</title>",
            self.css_style,
            "</head>",
            "<body>",
        ]

        # Header
        html_parts.extend(self._generate_header())

        # Executive Summary
        if summary_text:
            html_parts.extend(self._generate_summary_section(summary_text))

        # Key Metrics
        html_parts.extend(self._generate_metrics_section(metrics))

        # Visualizations
        if figures:
            html_parts.extend(self._generate_visualizations_section(figures))

        # Detailed Tables
        html_parts.extend(self._generate_tables_section(metrics))

        # Interpretation
        html_parts.extend(self._generate_interpretation_section(metrics))

        # Footer
        html_parts.extend(self._generate_footer())

        html_parts.extend(["</body>", "</html>"])

        html_content = "\n".join(html_parts)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        return html_content

    def generate_summary(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> str:
        """
        Generate a text summary of key findings.

        Args:
            metrics: Dictionary of computed metrics
            output_path: Optional path to save the summary

        Returns:
            Summary text
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append(f"{self.project_name} - Analysis Summary".center(80))
        lines.append("=" * 80)
        lines.append(f"Generated: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.author:
            lines.append(f"Author: {self.author}")
        lines.append("")

        # Temporal metrics
        if 'temporal' in metrics:
            lines.append("TEMPORAL METRICS")
            lines.append("-" * 80)
            temporal = metrics['temporal']

            if 'stride_time' in temporal:
                self._add_metric_summary(lines, "Stride Time", temporal['stride_time'], "s")
            if 'stride_length' in temporal:
                self._add_metric_summary(lines, "Stride Length", temporal['stride_length'], "m")
            if 'cadence' in temporal:
                self._add_metric_summary(lines, "Cadence", temporal['cadence'], "steps/min")
            if 'speed' in temporal:
                self._add_metric_summary(lines, "Speed", temporal['speed'], "m/s")
            lines.append("")

        # Kinematic metrics
        if 'kinematics' in metrics:
            lines.append("KINEMATIC METRICS")
            lines.append("-" * 80)
            kinematics = metrics['kinematics']

            for joint in ['hip', 'knee', 'ankle']:
                for side in ['left', 'right']:
                    key = f"{side}_{joint}_angle"
                    if key in kinematics:
                        self._add_metric_summary(
                            lines,
                            f"{side.title()} {joint.title()} Angle",
                            kinematics[key],
                            "°"
                        )
            lines.append("")

        # Symmetry metrics
        if 'symmetry' in metrics:
            lines.append("SYMMETRY ANALYSIS")
            lines.append("-" * 80)
            symmetry = metrics['symmetry']

            for metric_name, value in symmetry.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {metric_name.replace('_', ' ').title()}: {value:.2f}%")
                elif isinstance(value, dict) and 'mean' in value:
                    lines.append(f"  {metric_name.replace('_', ' ').title()}: {value['mean']:.2f}% ± {value.get('std', 0):.2f}%")
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            lines.append(f"  • {rec}")
        lines.append("")

        lines.append("=" * 80)

        summary_text = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)

        return summary_text

    def _generate_header(self) -> List[str]:
        """Generate HTML header section."""
        return [
            "<div class='header'>",
            f"<h1>{self.project_name}</h1>",
            f"<div class='subtitle'>Analysis Report</div>",
            f"<div class='subtitle'>Generated: {self.created_at.strftime('%B %d, %Y at %H:%M')}</div>",
            f"<div class='subtitle'>{f'Author: {self.author}' if self.author else ''}</div>",
            "</div>",
        ]

    def _generate_summary_section(self, summary_text: str) -> List[str]:
        """Generate executive summary section."""
        return [
            "<div class='section'>",
            "<h2>Executive Summary</h2>",
            f"<p>{summary_text}</p>",
            "</div>",
        ]

    def _generate_metrics_section(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate key metrics section with metric cards."""
        html = [
            "<div class='section'>",
            "<h2>Key Metrics</h2>",
            "<div class='metrics-grid'>",
        ]

        # Extract key metrics for display
        key_metrics = self._extract_key_metrics(metrics)

        for metric in key_metrics:
            html.extend([
                "<div class='metric-card'>",
                f"<div class='metric-label'>{metric['label']}</div>",
                f"<div class='metric-value'>{metric['value']}</div>",
                f"<div class='metric-unit'>{metric['unit']}</div>",
                "</div>",
            ])

        html.extend([
            "</div>",
            "</div>",
        ])

        return html

    def _generate_visualizations_section(
        self,
        figures: Dict[str, plt.Figure]
    ) -> List[str]:
        """Generate visualizations section with embedded figures."""
        html = [
            "<div class='section'>",
            "<h2>Visualizations</h2>",
        ]

        for title, fig in figures.items():
            # Convert figure to base64 image
            img_data = self._fig_to_base64(fig)

            html.extend([
                f"<h3>{title}</h3>",
                "<div class='plot-container'>",
                f"<img src='data:image/png;base64,{img_data}' alt='{title}'>",
                "</div>",
            ])

        html.append("</div>")

        return html

    def _generate_tables_section(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate detailed tables section."""
        html = [
            "<div class='section'>",
            "<h2>Detailed Metrics</h2>",
        ]

        # Generate tables for each metric category
        for category, data in metrics.items():
            if isinstance(data, dict):
                html.append(f"<h3>{category.replace('_', ' ').title()}</h3>")
                html.extend(self._dict_to_table(data))

        html.append("</div>")

        return html

    def _generate_interpretation_section(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate interpretation and recommendations section."""
        html = [
            "<div class='section'>",
            "<h2>Interpretation & Recommendations</h2>",
        ]

        # Generate recommendations based on metrics
        recommendations = self._generate_recommendations(metrics)

        for rec_type, message in recommendations:
            html.append(f"<div class='alert alert-{rec_type}'>{message}</div>")

        html.append("</div>")

        return html

    def _generate_footer(self) -> List[str]:
        """Generate HTML footer."""
        return [
            "<div class='footer'>",
            f"<p>Report generated by {self.project_name}</p>",
            f"<p>&copy; {self.created_at.year} - Running Gait Analysis System</p>",
            "</div>",
        ]

    def _extract_key_metrics(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract key metrics for card display."""
        key_metrics = []

        # Temporal metrics
        if 'temporal' in metrics:
            temporal = metrics['temporal']
            if 'cadence' in temporal and isinstance(temporal['cadence'], dict):
                key_metrics.append({
                    'label': 'Cadence',
                    'value': f"{temporal['cadence'].get('mean', 0):.1f}",
                    'unit': 'steps/min'
                })
            if 'stride_length' in temporal and isinstance(temporal['stride_length'], dict):
                key_metrics.append({
                    'label': 'Stride Length',
                    'value': f"{temporal['stride_length'].get('mean', 0):.2f}",
                    'unit': 'meters'
                })
            if 'speed' in temporal and isinstance(temporal['speed'], dict):
                key_metrics.append({
                    'label': 'Speed',
                    'value': f"{temporal['speed'].get('mean', 0):.2f}",
                    'unit': 'm/s'
                })

        # Symmetry
        if 'symmetry' in metrics:
            symmetry = metrics['symmetry']
            if 'overall_symmetry_index' in symmetry:
                val = symmetry['overall_symmetry_index']
                if isinstance(val, dict):
                    val = val.get('mean', 0)
                key_metrics.append({
                    'label': 'Symmetry Index',
                    'value': f"{val:.1f}",
                    'unit': '%'
                })

        return key_metrics

    def _dict_to_table(self, data: Dict[str, Any]) -> List[str]:
        """Convert a dictionary to an HTML table."""
        html = [
            "<table>",
            "<thead><tr><th>Metric</th><th>Value</th></tr></thead>",
            "<tbody>",
        ]

        for key, value in data.items():
            label = key.replace('_', ' ').title()

            if isinstance(value, dict):
                # Handle nested dictionary (e.g., mean, std, min, max)
                if 'mean' in value:
                    val_str = f"{value['mean']:.2f}"
                    if 'std' in value:
                        val_str += f" ± {value['std']:.2f}"
                    if 'min' in value and 'max' in value:
                        val_str += f" (range: {value['min']:.2f} - {value['max']:.2f})"
                else:
                    val_str = str(value)
            elif isinstance(value, (int, float)):
                val_str = f"{value:.2f}"
            else:
                val_str = str(value)

            html.append(f"<tr><td>{label}</td><td>{val_str}</td></tr>")

        html.extend([
            "</tbody>",
            "</table>",
        ])

        return html

    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Generate recommendations based on metrics."""
        recommendations = []

        # Check symmetry
        if 'symmetry' in metrics:
            symmetry = metrics['symmetry']
            if 'overall_symmetry_index' in symmetry:
                val = symmetry['overall_symmetry_index']
                if isinstance(val, dict):
                    val = val.get('mean', 100)

                if val < 90:
                    recommendations.append((
                        'warning',
                        f"Significant asymmetry detected ({val:.1f}%). Consider consulting with a "
                        "running coach or physical therapist to address potential imbalances."
                    ))
                elif val >= 95:
                    recommendations.append((
                        'success',
                        f"Excellent symmetry ({val:.1f}%). Your left and right sides are well balanced."
                    ))

        # Check cadence
        if 'temporal' in metrics and 'cadence' in metrics['temporal']:
            cadence = metrics['temporal']['cadence']
            if isinstance(cadence, dict):
                cadence_val = cadence.get('mean', 0)
            else:
                cadence_val = cadence

            if cadence_val < 160:
                recommendations.append((
                    'info',
                    f"Your cadence is {cadence_val:.0f} steps/min. Research suggests that increasing "
                    "cadence to 170-180 steps/min may reduce injury risk and improve efficiency."
                ))
            elif cadence_val >= 170:
                recommendations.append((
                    'success',
                    f"Your cadence ({cadence_val:.0f} steps/min) is within the optimal range for runners."
                ))

        # Default recommendation if none generated
        if not recommendations:
            recommendations.append((
                'info',
                "Continue monitoring your gait metrics to track progress and identify trends over time."
            ))

        return recommendations

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64

    def _add_metric_summary(
        self,
        lines: List[str],
        label: str,
        value: Any,
        unit: str
    ) -> None:
        """Add a metric summary line to the text report."""
        if isinstance(value, dict):
            if 'mean' in value:
                val_str = f"{value['mean']:.2f}"
                if 'std' in value:
                    val_str += f" ± {value['std']:.2f}"
                if 'min' in value and 'max' in value:
                    val_str += f" (range: {value['min']:.2f} - {value['max']:.2f})"
                lines.append(f"  {label}: {val_str} {unit}")
        elif isinstance(value, (int, float)):
            lines.append(f"  {label}: {value:.2f} {unit}")
