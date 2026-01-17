"""HTML Report Generator for QC metrics.

This module generates interactive HTML reports with visualizations
of QC metrics collected during preprocessing.

Features:
- Summary statistics section
- SNR/CNR metrics charts
- Pre-vs-post comparison charts
- Mask comparison metrics
- Outlier highlighting
- Interactive tables with sorting
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """Generates interactive HTML reports for QC metrics.

    This class builds a self-contained HTML file with embedded CSS and JavaScript
    (Chart.js for visualizations) that can be viewed in any web browser.
    """

    def __init__(
        self,
        title: str = "MenGrowth Preprocessing QC Report",
        output_dir: Optional[Path] = None
    ) -> None:
        """Initialize HTML report generator.

        Args:
            title: Report title
            output_dir: Directory for output files
        """
        self.title = title
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.sections: List[str] = []
        self._chart_counter = 0

    def _get_chart_id(self) -> str:
        """Generate unique chart ID."""
        self._chart_counter += 1
        return f"chart_{self._chart_counter}"

    def add_summary_section(self, summary: Dict[str, Any]) -> None:
        """Add summary statistics section.

        Args:
            summary: Dict with summary statistics
        """
        html = """
        <div class="section">
            <h2>Summary</h2>
            <div class="summary-grid">
        """

        # Key metrics as cards
        cards = [
            ("Total Records", summary.get("n_records", "N/A")),
            ("Pipeline Run ID", summary.get("pipeline_run_id", "N/A")[:8] + "..."),
            ("Timestamp", summary.get("timestamp", "N/A")),
            ("Reference Groups", len(summary.get("reference_groups", []))),
        ]

        for label, value in cards:
            html += f"""
                <div class="summary-card">
                    <div class="card-value">{value}</div>
                    <div class="card-label">{label}</div>
                </div>
            """

        html += """
            </div>
        </div>
        """

        self.sections.append(html)

    def add_snr_cnr_section(
        self,
        metrics_df: pd.DataFrame,
        group_by: str = "step_base"
    ) -> None:
        """Add SNR/CNR metrics section with charts.

        Args:
            metrics_df: DataFrame with QC metrics
            group_by: Column to group by for visualization
        """
        # Check if SNR/CNR columns exist
        snr_cols = [c for c in metrics_df.columns if 'snr' in c.lower()]
        cnr_cols = [c for c in metrics_df.columns if 'cnr' in c.lower()]

        if not snr_cols and not cnr_cols:
            return

        html = """
        <div class="section">
            <h2>SNR/CNR Metrics</h2>
        """

        # SNR chart
        if 'snr_background' in metrics_df.columns:
            chart_id = self._get_chart_id()
            snr_data = metrics_df.groupby(group_by)['snr_background'].mean().dropna()

            if len(snr_data) > 0:
                labels = list(snr_data.index)
                values = [float(v) for v in snr_data.values]

                html += f"""
                <div class="chart-container">
                    <h3>Background-based SNR by Step</h3>
                    <canvas id="{chart_id}"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('{chart_id}'), {{
                        type: 'bar',
                        data: {{
                            labels: {json.dumps(labels)},
                            datasets: [{{
                                label: 'Mean SNR',
                                data: {json.dumps(values)},
                                backgroundColor: 'rgba(54, 162, 235, 0.7)',
                                borderColor: 'rgba(54, 162, 235, 1)',
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'SNR' }} }}
                            }}
                        }}
                    }});
                </script>
                """

        # CNR chart
        if 'cnr_high_low' in metrics_df.columns:
            chart_id = self._get_chart_id()
            cnr_data = metrics_df.groupby(group_by)['cnr_high_low'].mean().dropna()

            if len(cnr_data) > 0:
                labels = list(cnr_data.index)
                values = [float(v) for v in cnr_data.values]

                html += f"""
                <div class="chart-container">
                    <h3>CNR (High-Low) by Step</h3>
                    <canvas id="{chart_id}"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('{chart_id}'), {{
                        type: 'bar',
                        data: {{
                            labels: {json.dumps(labels)},
                            datasets: [{{
                                label: 'Mean CNR',
                                data: {json.dumps(values)},
                                backgroundColor: 'rgba(75, 192, 192, 0.7)',
                                borderColor: 'rgba(75, 192, 192, 1)',
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            scales: {{
                                y: {{ beginAtZero: true, title: {{ display: true, text: 'CNR' }} }}
                            }}
                        }}
                    }});
                </script>
                """

        html += "</div>"
        self.sections.append(html)

    def add_pre_post_comparison_section(
        self,
        metrics_df: pd.DataFrame
    ) -> None:
        """Add pre-vs-post comparison section.

        Args:
            metrics_df: DataFrame with QC metrics including comparison columns
        """
        # Find comparison columns (those ending in _delta or _ratio)
        delta_cols = [c for c in metrics_df.columns if c.endswith('_delta')]
        ratio_cols = [c for c in metrics_df.columns if c.endswith('_ratio')]

        if not delta_cols and not ratio_cols:
            return

        html = """
        <div class="section">
            <h2>Pre vs Post Comparison</h2>
        """

        # Ratio chart (most useful for seeing improvement/degradation)
        if ratio_cols:
            chart_id = self._get_chart_id()

            # Aggregate ratios by step
            ratio_data = {}
            for col in ratio_cols[:5]:  # Limit to 5 most important
                metric_name = col.replace('_ratio', '')
                mean_ratio = metrics_df[col].dropna().mean()
                if not np.isnan(mean_ratio):
                    ratio_data[metric_name] = mean_ratio

            if ratio_data:
                labels = list(ratio_data.keys())
                values = list(ratio_data.values())

                html += f"""
                <div class="chart-container">
                    <h3>Metric Ratios (Post/Baseline)</h3>
                    <p class="chart-note">Values > 1 indicate improvement; &lt; 1 indicate degradation</p>
                    <canvas id="{chart_id}"></canvas>
                </div>
                <script>
                    new Chart(document.getElementById('{chart_id}'), {{
                        type: 'bar',
                        data: {{
                            labels: {json.dumps(labels)},
                            datasets: [{{
                                label: 'Ratio (Post/Baseline)',
                                data: {json.dumps(values)},
                                backgroundColor: {json.dumps([
                                    'rgba(75, 192, 192, 0.7)' if v >= 1 else 'rgba(255, 99, 132, 0.7)'
                                    for v in values
                                ])},
                                borderColor: {json.dumps([
                                    'rgba(75, 192, 192, 1)' if v >= 1 else 'rgba(255, 99, 132, 1)'
                                    for v in values
                                ])},
                                borderWidth: 1
                            }}]
                        }},
                        options: {{
                            responsive: true,
                            scales: {{
                                y: {{
                                    beginAtZero: false,
                                    title: {{ display: true, text: 'Ratio' }},
                                    suggestedMin: 0.5,
                                    suggestedMax: 1.5
                                }}
                            }},
                            plugins: {{
                                annotation: {{
                                    annotations: {{
                                        line1: {{
                                            type: 'line',
                                            yMin: 1,
                                            yMax: 1,
                                            borderColor: 'rgb(100, 100, 100)',
                                            borderWidth: 2,
                                            borderDash: [5, 5]
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }});
                </script>
                """

        html += "</div>"
        self.sections.append(html)

    def add_mask_comparison_section(
        self,
        mask_comparisons: Dict[str, Any]
    ) -> None:
        """Add mask comparison section.

        Args:
            mask_comparisons: Dict with mask comparison results from QC manager
        """
        if not mask_comparisons:
            return

        html = """
        <div class="section">
            <h2>Longitudinal Mask Comparison</h2>
        """

        # Collect all Dice scores
        dice_scores = []
        patient_labels = []

        for patient_id, patient_data in mask_comparisons.items():
            summary = patient_data.get("summary", {})
            dice_mean = summary.get("dice_mean")
            if dice_mean is not None:
                dice_scores.append(dice_mean)
                patient_labels.append(patient_id)

        if dice_scores:
            chart_id = self._get_chart_id()

            html += f"""
            <div class="chart-container">
                <h3>Mean Dice Score by Patient</h3>
                <p class="chart-note">Higher is better (1.0 = perfect overlap)</p>
                <canvas id="{chart_id}"></canvas>
            </div>
            <script>
                new Chart(document.getElementById('{chart_id}'), {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(patient_labels)},
                        datasets: [{{
                            label: 'Mean Dice Score',
                            data: {json.dumps(dice_scores)},
                            backgroundColor: {json.dumps([
                                'rgba(75, 192, 192, 0.7)' if d >= 0.85 else 'rgba(255, 206, 86, 0.7)' if d >= 0.7 else 'rgba(255, 99, 132, 0.7)'
                                for d in dice_scores
                            ])},
                            borderColor: {json.dumps([
                                'rgba(75, 192, 192, 1)' if d >= 0.85 else 'rgba(255, 206, 86, 1)' if d >= 0.7 else 'rgba(255, 99, 132, 1)'
                                for d in dice_scores
                            ])},
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        scales: {{
                            y: {{
                                beginAtZero: false,
                                min: 0,
                                max: 1,
                                title: {{ display: true, text: 'Dice Score' }}
                            }}
                        }}
                    }}
                }});
            </script>
            """

            # Summary table
            html += """
            <div class="table-container">
                <h3>Summary Statistics</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Patient</th>
                            <th>N Comparisons</th>
                            <th>Mean Dice</th>
                            <th>Min Dice</th>
                            <th>Max Dice</th>
                            <th>Low Dice Count</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for patient_id, patient_data in mask_comparisons.items():
                summary = patient_data.get("summary", {})
                html += f"""
                    <tr>
                        <td>{patient_id}</td>
                        <td>{summary.get('n_comparisons', 'N/A')}</td>
                        <td>{summary.get('dice_mean', 'N/A'):.3f if summary.get('dice_mean') else 'N/A'}</td>
                        <td>{summary.get('dice_min', 'N/A'):.3f if summary.get('dice_min') else 'N/A'}</td>
                        <td>{summary.get('dice_max', 'N/A'):.3f if summary.get('dice_max') else 'N/A'}</td>
                        <td class="{'warning' if summary.get('n_low_dice', 0) > 0 else ''}">{summary.get('n_low_dice', 0)}</td>
                    </tr>
                """

            html += """
                    </tbody>
                </table>
            </div>
            """

        html += "</div>"
        self.sections.append(html)

    def add_outlier_section(
        self,
        metrics_df: pd.DataFrame
    ) -> None:
        """Add outlier detection section.

        Args:
            metrics_df: DataFrame with QC metrics including outlier flags
        """
        # Find outlier columns
        outlier_cols = [c for c in metrics_df.columns if c.endswith('_outlier')]

        if not outlier_cols:
            return

        html = """
        <div class="section">
            <h2>Outlier Detection</h2>
        """

        # Count outliers per metric
        outlier_counts = {}
        for col in outlier_cols:
            metric_name = col.replace('_outlier', '')
            count = metrics_df[col].sum() if col in metrics_df.columns else 0
            if count > 0:
                outlier_counts[metric_name] = int(count)

        if outlier_counts:
            chart_id = self._get_chart_id()
            labels = list(outlier_counts.keys())
            values = list(outlier_counts.values())

            html += f"""
            <div class="chart-container">
                <h3>Outlier Counts by Metric</h3>
                <canvas id="{chart_id}"></canvas>
            </div>
            <script>
                new Chart(document.getElementById('{chart_id}'), {{
                    type: 'bar',
                    data: {{
                        labels: {json.dumps(labels)},
                        datasets: [{{
                            label: 'Outlier Count',
                            data: {json.dumps(values)},
                            backgroundColor: 'rgba(255, 99, 132, 0.7)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        indexAxis: 'y',
                        scales: {{
                            x: {{ beginAtZero: true, title: {{ display: true, text: 'Count' }} }}
                        }}
                    }}
                }});
            </script>
            """

        # Table of actual outliers
        outlier_rows = []
        for _, row in metrics_df.iterrows():
            for col in outlier_cols:
                if row.get(col, False):
                    metric_name = col.replace('_outlier', '')
                    outlier_rows.append({
                        'patient_id': row.get('patient_id', 'N/A'),
                        'study_id': row.get('study_id', 'N/A'),
                        'modality': row.get('modality', 'N/A'),
                        'step': row.get('step_base', 'N/A'),
                        'metric': metric_name,
                        'value': row.get(metric_name, 'N/A')
                    })

        if outlier_rows:
            html += """
            <div class="table-container">
                <h3>Detected Outliers</h3>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Patient</th>
                            <th>Study</th>
                            <th>Modality</th>
                            <th>Step</th>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
            """

            for outlier in outlier_rows[:50]:  # Limit to 50 rows
                val = outlier['value']
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                html += f"""
                    <tr class="outlier-row">
                        <td>{outlier['patient_id']}</td>
                        <td>{outlier['study_id']}</td>
                        <td>{outlier['modality']}</td>
                        <td>{outlier['step']}</td>
                        <td>{outlier['metric']}</td>
                        <td>{val_str}</td>
                    </tr>
                """

            html += """
                    </tbody>
                </table>
            </div>
            """

        html += "</div>"
        self.sections.append(html)

    def add_metrics_table_section(
        self,
        metrics_df: pd.DataFrame,
        max_rows: int = 100
    ) -> None:
        """Add interactive metrics table section.

        Args:
            metrics_df: DataFrame with QC metrics
            max_rows: Maximum rows to display
        """
        # Select key columns for display
        display_cols = ['patient_id', 'study_id', 'modality', 'step_base', 'status']

        # Add numeric columns (excluding internal ones)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if not c.startswith('_') and not c.endswith('_outlier')]
        display_cols.extend(numeric_cols[:10])  # Limit to 10 numeric columns

        # Filter to existing columns
        display_cols = [c for c in display_cols if c in metrics_df.columns]

        if not display_cols:
            return

        html = """
        <div class="section">
            <h2>Metrics Data</h2>
            <div class="table-container" style="overflow-x: auto;">
                <table class="data-table sortable">
                    <thead>
                        <tr>
        """

        for col in display_cols:
            html += f"<th onclick=\"sortTable(this)\">{col}</th>"

        html += """
                        </tr>
                    </thead>
                    <tbody>
        """

        for _, row in metrics_df.head(max_rows).iterrows():
            status = row.get('status', '')
            row_class = 'error-row' if status == 'error' else ''
            html += f"<tr class=\"{row_class}\">"
            for col in display_cols:
                val = row.get(col, '')
                if isinstance(val, float):
                    val = f"{val:.4f}"
                html += f"<td>{val}</td>"
            html += "</tr>"

        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """

        self.sections.append(html)

    def generate(self, output_path: Optional[Path] = None) -> Path:
        """Generate the HTML report.

        Args:
            output_path: Optional specific output path (default: output_dir/qc_report.html)

        Returns:
            Path to generated HTML file
        """
        if output_path is None:
            output_path = self.output_dir / "qc_report.html"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self._get_html_template()

        # Insert sections
        sections_html = "\n".join(self.sections)
        html = html.replace("<!-- SECTIONS -->", sections_html)

        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Generated HTML report: {output_path}")
        return output_path

    def _get_html_template(self) -> str:
        """Get HTML template with embedded CSS and JavaScript."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --bg-color: #f5f6fa;
            --card-bg: #ffffff;
            --text-color: #2c3e50;
            --border-color: #dcdde1;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 10px;
        }}

        header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        header .timestamp {{
            font-size: 0.9rem;
            opacity: 0.8;
        }}

        .section {{
            background: var(--card-bg);
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .section h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid var(--secondary-color);
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .section h3 {{
            color: var(--primary-color);
            margin: 20px 0 15px 0;
            font-size: 1.1rem;
        }}

        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .summary-card {{
            background: var(--bg-color);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}

        .card-value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: var(--secondary-color);
        }}

        .card-label {{
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 5px;
        }}

        .chart-container {{
            margin: 20px 0;
            padding: 15px;
            background: var(--bg-color);
            border-radius: 8px;
        }}

        .chart-note {{
            font-size: 0.85rem;
            color: #7f8c8d;
            margin-bottom: 10px;
        }}

        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        .data-table th,
        .data-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}

        .data-table th {{
            background: var(--primary-color);
            color: white;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
        }}

        .data-table th:hover {{
            background: var(--secondary-color);
        }}

        .data-table tr:hover {{
            background: var(--bg-color);
        }}

        .data-table .outlier-row {{
            background: rgba(231, 76, 60, 0.1);
        }}

        .data-table .error-row {{
            background: rgba(231, 76, 60, 0.2);
        }}

        .data-table .warning {{
            color: var(--warning-color);
            font-weight: bold;
        }}

        footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            font-size: 0.85rem;
        }}

        @media (max-width: 768px) {{
            .container {{
                padding: 10px;
            }}

            header {{
                padding: 20px;
            }}

            header h1 {{
                font-size: 1.5rem;
            }}

            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{self.title}</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </header>

        <!-- SECTIONS -->

        <footer>
            <p>Generated by MenGrowth QC Pipeline</p>
        </footer>
    </div>

    <script>
        function sortTable(header) {{
            const table = header.closest('table');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const columnIndex = Array.from(header.parentNode.children).indexOf(header);
            const isNumeric = rows.some(row => !isNaN(parseFloat(row.children[columnIndex]?.textContent)));

            const direction = header.dataset.sortDir === 'asc' ? 'desc' : 'asc';
            header.dataset.sortDir = direction;

            rows.sort((a, b) => {{
                const aVal = a.children[columnIndex]?.textContent || '';
                const bVal = b.children[columnIndex]?.textContent || '';

                if (isNumeric) {{
                    const aNum = parseFloat(aVal) || 0;
                    const bNum = parseFloat(bVal) || 0;
                    return direction === 'asc' ? aNum - bNum : bNum - aNum;
                }} else {{
                    return direction === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                }}
            }});

            rows.forEach(row => tbody.appendChild(row));
        }}
    </script>
</body>
</html>"""


def generate_qc_report(
    metrics_df: pd.DataFrame,
    summary: Dict[str, Any],
    mask_comparisons: Optional[Dict[str, Any]] = None,
    output_dir: Path = Path("."),
    title: str = "MenGrowth Preprocessing QC Report"
) -> Path:
    """Convenience function to generate a complete QC report.

    Args:
        metrics_df: DataFrame with QC metrics
        summary: Summary statistics dict
        mask_comparisons: Optional mask comparison results
        output_dir: Output directory for report
        title: Report title

    Returns:
        Path to generated HTML report
    """
    generator = HTMLReportGenerator(title=title, output_dir=output_dir)

    # Add sections
    generator.add_summary_section(summary)
    generator.add_snr_cnr_section(metrics_df)
    generator.add_pre_post_comparison_section(metrics_df)

    if mask_comparisons:
        generator.add_mask_comparison_section(mask_comparisons)

    generator.add_outlier_section(metrics_df)
    generator.add_metrics_table_section(metrics_df)

    return generator.generate()
