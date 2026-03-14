"""
Plotting and visualization for gait analysis metrics.

This module provides classes for creating publication-quality plots of gait metrics
using matplotlib and seaborn.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


class MetricsPlotter:
    """
    Class for creating plots of gait analysis metrics.

    This class provides methods to visualize angles over time, gait cycles,
    symmetry metrics, and other biomechanical measurements.
    """

    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
    ):
        """
        Initialize the MetricsPlotter.

        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
            dpi: Resolution for saved figures
        """
        self.figsize = figsize
        self.dpi = dpi

        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')

        # Set seaborn color palette
        sns.set_palette("husl")

        # Define color scheme for body sides
        self.side_colors = {
            'left': '#2E86AB',    # Blue
            'right': '#A23B72',   # Purple/Red
        }

        # Define colors for different angles
        self.angle_colors = {
            'hip': '#E63946',
            'knee': '#2A9D8F',
            'ankle': '#E9C46A',
            'trunk': '#264653',
        }

    def plot_angles_over_time(
        self,
        data: pd.DataFrame,
        angles: List[str],
        time_column: str = 'time',
        title: str = "Joint Angles Over Time",
        xlabel: str = "Time (s)",
        ylabel: str = "Angle (degrees)",
        save_path: Optional[Path] = None,
        show_events: bool = True,
        events_data: Optional[pd.DataFrame] = None,
    ) -> plt.Figure:
        """
        Plot joint angles over time.

        Args:
            data: DataFrame containing angle measurements
            angles: List of angle column names to plot
            time_column: Name of the time column
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save the figure
            show_events: Whether to show gait events
            events_data: Optional DataFrame with gait event timings

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot each angle
        for angle in angles:
            if angle not in data.columns:
                print(f"Warning: Angle '{angle}' not found in data")
                continue

            # Determine color based on angle name
            color = None
            for joint_name, joint_color in self.angle_colors.items():
                if joint_name in angle.lower():
                    color = joint_color
                    break

            # Determine line style based on side (left/right)
            linestyle = '-'
            if 'left' in angle.lower():
                linestyle = '-'
            elif 'right' in angle.lower():
                linestyle = '--'

            ax.plot(
                data[time_column],
                data[angle],
                label=angle.replace('_', ' ').title(),
                linewidth=2,
                color=color,
                linestyle=linestyle,
            )

        # Add gait events if provided
        if show_events and events_data is not None:
            self._add_gait_events_to_plot(ax, events_data, data[time_column].min(),
                                         data[time_column].max())

        # Formatting
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_gait_cycle(
        self,
        data: pd.DataFrame,
        angle_columns: List[str],
        cycle_percentage_column: str = 'cycle_percent',
        title: str = "Gait Cycle Analysis",
        save_path: Optional[Path] = None,
        show_phases: bool = True,
    ) -> plt.Figure:
        """
        Plot joint angles across a normalized gait cycle (0-100%).

        Args:
            data: DataFrame with gait cycle data
            angle_columns: List of angle column names to plot
            cycle_percentage_column: Name of the cycle percentage column
            title: Plot title
            save_path: Optional path to save the figure
            show_phases: Whether to show gait phase regions

        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(len(angle_columns), 1,
                                figsize=(self.figsize[0], 4 * len(angle_columns)),
                                dpi=self.dpi)

        if len(angle_columns) == 1:
            axes = [axes]

        for idx, angle in enumerate(angle_columns):
            ax = axes[idx]

            if angle not in data.columns:
                print(f"Warning: Angle '{angle}' not found in data")
                continue

            # Determine side from column name
            side = 'left' if 'left' in angle.lower() else 'right' if 'right' in angle.lower() else None

            # Plot angle
            color = self.side_colors.get(side, '#333333') if side else '#333333'

            # Group by cycle if multiple cycles exist
            if 'cycle_id' in data.columns:
                for cycle_id in data['cycle_id'].unique():
                    cycle_data = data[data['cycle_id'] == cycle_id]
                    ax.plot(
                        cycle_data[cycle_percentage_column],
                        cycle_data[angle],
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                    )

                # Plot mean
                mean_data = data.groupby(cycle_percentage_column)[angle].mean()
                ax.plot(
                    mean_data.index,
                    mean_data.values,
                    color=color,
                    linewidth=3,
                    label='Mean',
                )

                # Add standard deviation bands
                std_data = data.groupby(cycle_percentage_column)[angle].std()
                ax.fill_between(
                    mean_data.index,
                    mean_data.values - std_data.values,
                    mean_data.values + std_data.values,
                    color=color,
                    alpha=0.2,
                    label='±1 SD',
                )
            else:
                ax.plot(
                    data[cycle_percentage_column],
                    data[angle],
                    color=color,
                    linewidth=2,
                )

            # Add gait phase regions
            if show_phases:
                self._add_gait_phases(ax)

            # Formatting
            ax.set_ylabel(f"{angle.replace('_', ' ').title()}\n(degrees)",
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', framealpha=0.9)

            if idx == 0:
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

            if idx == len(angle_columns) - 1:
                ax.set_xlabel("Gait Cycle (%)", fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_symmetry(
        self,
        data: pd.DataFrame,
        left_column: str,
        right_column: str,
        title: str = "Left vs Right Symmetry",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot symmetry comparison between left and right sides.

        Args:
            data: DataFrame containing left and right measurements
            left_column: Name of left side column
            right_column: Name of right side column
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs = GridSpec(2, 2, figure=fig)

        # Subplot 1: Time series comparison
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data.index, data[left_column], label='Left',
                color=self.side_colors['left'], linewidth=2)
        ax1.plot(data.index, data[right_column], label='Right',
                color=self.side_colors['right'], linewidth=2, linestyle='--')
        ax1.set_ylabel("Angle (degrees)", fontsize=11, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(data[left_column], data[right_column],
                   alpha=0.5, s=30, edgecolors='black', linewidth=0.5)

        # Add line of identity
        min_val = min(data[left_column].min(), data[right_column].min())
        max_val = max(data[left_column].max(), data[right_column].max())
        ax2.plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect symmetry')

        ax2.set_xlabel("Left (degrees)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Right (degrees)", fontsize=11, fontweight='bold')
        ax2.set_title("Correlation Plot", fontsize=12, fontweight='bold')
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')

        # Subplot 3: Difference histogram
        ax3 = fig.add_subplot(gs[1, 1])
        difference = data[left_column] - data[right_column]
        ax3.hist(difference, bins=30, color='gray', alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
        ax3.axvline(difference.mean(), color='blue', linestyle='-',
                   linewidth=2, label=f'Mean: {difference.mean():.2f}°')

        ax3.set_xlabel("Difference (Left - Right, degrees)", fontsize=11, fontweight='bold')
        ax3.set_ylabel("Frequency", fontsize=11, fontweight='bold')
        ax3.set_title("Asymmetry Distribution", fontsize=12, fontweight='bold')
        ax3.legend(loc='best', framealpha=0.9)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_stride_metrics(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Stride Metrics",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot stride-level metrics (stride length, stride time, cadence, etc.).

        Args:
            metrics: Dictionary with metric names as keys and lists of values
            title: Plot title
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5), dpi=self.dpi)

        if n_metrics == 1:
            axes = [axes]

        for idx, (metric_name, values) in enumerate(metrics.items()):
            ax = axes[idx]

            # Create box plot with individual points
            bp = ax.boxplot([values], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))

            # Overlay individual points
            x = np.random.normal(1, 0.04, size=len(values))
            ax.scatter(x, values, alpha=0.5, s=50, edgecolors='black', linewidth=0.5)

            # Add mean line
            mean_val = np.mean(values)
            ax.plot([0.7, 1.3], [mean_val, mean_val], 'g--', linewidth=2, label=f'Mean: {mean_val:.2f}')

            # Add statistics text
            stats_text = f"Mean: {mean_val:.2f}\nSD: {np.std(values):.2f}\nCV: {(np.std(values)/mean_val*100):.1f}%"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

            # Formatting
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_xticks([])
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Heatmap",
        save_path: Optional[Path] = None,
        cmap: str = "coolwarm",
    ) -> plt.Figure:
        """
        Plot correlation heatmap of gait metrics.

        Args:
            data: DataFrame with metrics as columns
            title: Plot title
            save_path: Optional path to save the figure
            cmap: Colormap to use

        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Calculate correlation matrix
        corr_matrix = data.corr()

        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def plot_comparison(
        self,
        data_dict: Dict[str, pd.DataFrame],
        metric: str,
        title: str = "Metric Comparison",
        ylabel: str = "Value",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Compare a metric across multiple conditions/sessions.

        Args:
            data_dict: Dictionary with condition names as keys and DataFrames as values
            metric: Name of the metric column to compare
            title: Plot title
            ylabel: Y-axis label
            save_path: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)

        # Prepare data for plotting
        plot_data = []
        for condition, df in data_dict.items():
            if metric in df.columns:
                for value in df[metric]:
                    plot_data.append({'Condition': condition, 'Value': value})

        plot_df = pd.DataFrame(plot_data)

        # Violin plot
        sns.violinplot(data=plot_df, x='Condition', y='Value', ax=ax1)
        ax1.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax1.set_xlabel("Condition", fontsize=11, fontweight='bold')
        ax1.set_title("Distribution Comparison", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Box plot with points
        sns.boxplot(data=plot_df, x='Condition', y='Value', ax=ax2)
        sns.swarmplot(data=plot_df, x='Condition', y='Value', color='black',
                     alpha=0.5, size=3, ax=ax2)
        ax2.set_ylabel(ylabel, fontsize=11, fontweight='bold')
        ax2.set_xlabel("Condition", fontsize=11, fontweight='bold')
        ax2.set_title("Box Plot with Individual Values", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        return fig

    def _add_gait_phases(self, ax: plt.Axes) -> None:
        """
        Add gait phase regions to a plot (stance and swing phases).

        Args:
            ax: Matplotlib axes object
        """
        # Typical gait phases (approximate percentages)
        stance_phase = (0, 60)
        swing_phase = (60, 100)

        # Add shaded regions
        ax.axvspan(stance_phase[0], stance_phase[1],
                  alpha=0.1, color='green', label='Stance')
        ax.axvspan(swing_phase[0], swing_phase[1],
                  alpha=0.1, color='blue', label='Swing')

        # Add vertical line at toe-off (transition)
        ax.axvline(x=60, color='red', linestyle='--', linewidth=1.5,
                  alpha=0.7, label='Toe-off')

    def _add_gait_events_to_plot(
        self,
        ax: plt.Axes,
        events_data: pd.DataFrame,
        x_min: float,
        x_max: float,
    ) -> None:
        """
        Add gait event markers to a time series plot.

        Args:
            ax: Matplotlib axes object
            events_data: DataFrame with event timings and types
            x_min: Minimum x value for plot
            x_max: Maximum x value for plot
        """
        event_colors = {
            'foot_strike': 'red',
            'toe_off': 'blue',
            'mid_stance': 'green',
            'mid_swing': 'orange',
        }

        event_styles = {
            'left': '-',
            'right': '--',
        }

        for _, event in events_data.iterrows():
            if 'time' not in event or 'event_type' not in event:
                continue

            time = event['time']
            event_type = event['event_type']
            side = event.get('side', 'left')

            if time < x_min or time > x_max:
                continue

            color = event_colors.get(event_type, 'gray')
            linestyle = event_styles.get(side, '-')

            ax.axvline(x=time, color=color, linestyle=linestyle,
                      linewidth=1.5, alpha=0.7)

        # Create legend for events
        patches = []
        for event_type, color in event_colors.items():
            patches.append(mpatches.Patch(color=color, label=event_type.replace('_', ' ').title()))

        # Add to existing legend
        handles, labels = ax.get_legend_handles_labels()
        handles.extend(patches)
        ax.legend(handles=handles, loc='best', framealpha=0.9)

    def close_all(self) -> None:
        """Close all open figures."""
        plt.close('all')
