# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 00:20:33 2024

@author: Alexey
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import CA3D_functions as CA


def read_histogram_data(file_path):
    """
    Read the histogram data from the specified CSV file.
    """
    timesteps = []
    size_bins = []
    hist_data = []

    with open(file_path, 'r') as f:
        # Read the header
        header = f.readline().strip().split(", ")
        size_bins = [int(size.split("_")[1]) for size in header[1:]]  # Size_x
        
        # Read the data (skip header line)
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                values = list(map(float, line.split(", ")))
                timesteps.append(int(values[0]))  # First column is timestep
                hist_data.append(values[1:])  # Exclude timestep
            except ValueError as e:
                print(f"Warning: Skipping invalid line: {line} (Error: {e})")
                continue

    return timesteps, size_bins, np.array(hist_data)

def plot_histogram_data_v2(timesteps, size_bins, hist_data, output_prefix, 
                       min_point_size=3, max_point_size=70, 
                       alpha=0.3, log_scale=True):
    """
    Plot the histogram data as a scatter plot with point sizes proportional to counts.
    
    Parameters:
    -----------
    timesteps : array-like
        Array of timestep values
    size_bins : array-like
        Array of cluster size bin values
    hist_data : 2D array
        2D histogram data with shape (len(timesteps), len(size_bins))
    output_prefix : str
        Prefix for the output file name
    min_point_size : float
        Minimum point size in the scatter plot
    max_point_size : float
        Maximum point size in the scatter plot
    alpha : float
        Transparency of points (0 to 1)
    log_scale : bool
        Whether to use log scale for point sizes and colormap
    """
    # Create meshgrid for all combinations of timesteps and size_bins
    T, S = np.meshgrid(np.arange(len(timesteps)), np.arange(len(size_bins)))
    
    # Flatten the arrays for scatter plot
    t_flat = T.flatten()
    s_flat = S.flatten()
    counts_flat = hist_data.T.flatten()
    
    # Remove zero counts to avoid plotting points where there's no data
    mask = counts_flat > 0
    t_plot = t_flat[mask]
    s_plot = s_flat[mask]
    counts_plot = counts_flat[mask]
    
    # Create figure
    plt.figure(figsize=(35, 22))
    
    # Calculate point sizes
    if log_scale:
        sizes = np.interp(np.log10(counts_plot), 
                         (np.log10(counts_plot.min()), np.log10(counts_plot.max())), 
                         (min_point_size, max_point_size))
        norm = LogNorm(vmin=counts_plot.min(), vmax=counts_plot.max())
    else:
        sizes = np.interp(counts_plot, 
                         (counts_plot.min(), counts_plot.max()), 
                         (min_point_size, max_point_size))
        norm = plt.Normalize(vmin=counts_plot.min(), vmax=counts_plot.max())
    
    # Create scatter plot
    scatter = plt.scatter(t_plot, s_plot, 
                         c=counts_plot,
                         s=sizes,
                         alpha=alpha,
                         norm=norm,
                         cmap='viridis')
    
    # Customize ticks
    timestep_ticks = range(0, len(timesteps), max(1, len(timesteps) // 50))
    plt.xticks(timestep_ticks, [timesteps[i] for i in timestep_ticks], rotation=45)
    plt.xscale('log')
    size_ticks = range(0, len(size_bins), max(1, len(size_bins) // 50))
    plt.yticks(size_ticks, [size_bins[i] for i in size_ticks], rotation=45)
    
    # Labels and title
    plt.xlabel('Timestep')
    plt.ylabel('Cluster Size')
    plt.title('Cluster Size Distribution Evolution')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Number of Clusters')
    if log_scale:
        cbar.ax.set_yscale('log')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save and close
    plt.savefig(f'{output_prefix}_size_distribution_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_histogram_data_v3(timesteps, size_bins, hist_data, output_prefix, 
                          min_point_size=3, max_point_size=70, 
                          alpha=0.5, log_scale=False):
    """
    Plot the histogram data with logarithmic scaling and mean size evolution.
    
    Parameters:
    -----------
    timesteps : array-like
        Array of timestep values
    size_bins : array-like
        Array of cluster size bin values
    hist_data : 2D array
        2D histogram data with shape (len(timesteps), len(size_bins))
    output_prefix : str
        Prefix for the output file name
    min_point_size : float
        Minimum point size in the scatter plot
    max_point_size : float
        Maximum point size in the scatter plot
    alpha : float
        Transparency of points (0 to 1)
    log_scale : bool
        Whether to use log scale for point sizes and colormap
    """
    # Calculate mean size for each timestep
    total_clusters = hist_data.sum(axis=1)
    mean_size = np.zeros_like(total_clusters, dtype=float)
    plt.rcParams.update({
    'font.size': 24,          # Base font size
    'axes.titlesize': 28,     # Title font size
    'axes.labelsize': 26,     # Axis label size
    'xtick.labelsize': 24,    # X-tick label size
    'ytick.labelsize': 24,    # Y-tick label size
    'legend.fontsize': 24,    # Legend font size
    })

    for i in range(len(timesteps)):
        if total_clusters[i] > 0:
            mean_size[i] = np.sum(hist_data[i] * size_bins) / total_clusters[i]
    
    # Create meshgrid for all combinations of timesteps and size_bins
    T, S = np.meshgrid(timesteps, size_bins)
    
    # Flatten the arrays for scatter plot
    t_flat = T.flatten()
    s_flat = S.flatten()
    counts_flat = hist_data.T.flatten()
    
    # Remove zero counts
    mask = counts_flat > 0
    t_plot = t_flat[mask]
    s_plot = s_flat[mask]
    counts_plot = counts_flat[mask]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(35, 22))
    
    # Calculate point sizes
    if log_scale:
        sizes = np.interp(np.log10(counts_plot), 
                         (np.log10(counts_plot.min()), np.log10(counts_plot.max())), 
                         (min_point_size, max_point_size))
        norm = LogNorm(vmin=counts_plot.min(), vmax=counts_plot.max())
    else:
        sizes = np.interp(counts_plot, 
                         (counts_plot.min(), counts_plot.max()), 
                         (min_point_size, max_point_size))
        norm = plt.Normalize(vmin=counts_plot.min(), vmax=counts_plot.max())
    
    # Create scatter plot using the actual timesteps and size values
    scatter = ax.scatter(t_plot, s_plot,
                        c=counts_plot,
                        s=sizes,
                        alpha=alpha,
                        norm=norm,
                        cmap='viridis',
                        label='Cluster Distribution')
    
    # Plot mean size evolution on the same axis
    mean_line = ax.plot(timesteps[11:], mean_size[11:], 'r-', 
                       linewidth=6, label='Mean Size')
    
    # Set logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set x-axis limits to start from 10
    ax.set_xlim(left=10, right=max(timesteps))
    
    # Generate evenly spaced ticks for each decade
    def generate_decade_ticks(start_decade, end_decade, points_per_decade=1):
        ticks = []
        for decade in range(start_decade, end_decade + 1):
            base = 10 ** decade
            # Generate evenly spaced points within this decade
            decade_points = np.linspace(base, base * 10, points_per_decade, endpoint=False)
            ticks.extend(decade_points)
        return np.array(ticks)
    
    # Calculate start and end decades
    start_decade = int(np.floor(np.log10(10)))  # Start from 10
    end_decade = int(np.ceil(np.log10(max(timesteps)))-1)
    
    # Generate ticks
    #major_ticks = generate_decade_ticks(start_decade, end_decade)
    #ax.set_xticks(major_ticks)
    #ax.set_xticklabels([f'{int(x)}' for x in major_ticks], rotation=45)
    
    # Add minor ticks for better visual reference
    #ax.minorticks_on()
    
    # Set custom y-axis ticks for logarithmic scale
    y_ticks = [ 10, 100, 1000, 10000,100000]  # Adjust this list as needed
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    
    # Labels and title
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cluster Size')
    plt.title('Cluster Size Distribution and Mean Size Evolution')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Number of Clusters')
    if log_scale:
        cbar.ax.set_yscale('log')
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Add grid with logarithmic spacing
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and close
    plt.savefig(f'{output_prefix}_size_distribution_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    return mean_size

def read_alpha_data(file_path):
    """
    Read the alpha data from the specified file.
    Returns arrays of timesteps and alpha values.
    """
    timesteps = []
    alpha_values = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comment lines
            if not line or line.startswith('#'):
                continue
            
            try:
                values = list(map(float, line.split()))
                if len(values) >= 4:  # Ensure we have at least 4 columns
                    timesteps.append(values[0])
                    alpha_values.append(values[3])  # Fourth column contains alpha
            except ValueError as e:
                print(f"Warning: Skipping invalid line in alpha file: {line} (Error: {e})")
                continue
            
    return np.array(timesteps), np.array(alpha_values)

def plot_histogram_data_v4(timesteps, size_bins, hist_data, alpha_file_path, output_prefix, 
                          min_point_size=20, max_point_size=200, 
                          alpha=0.8, log_scale=False, t1=311100, single_plot=True):
    """
    Plot the histogram data with multiple y-axes showing:
    - Cluster size distribution (scatter)
    - Mean cluster size
    - Total number of clusters
    - Alpha parameter
    """
    # Calculate mean size and total clusters for each timestep
    snapshots = sorted(list(set(CA.generate_log_points(N=2000))))
    total_clusters = hist_data.sum(axis=1)
    mean_size = np.zeros_like(total_clusters, dtype=float)
    
    for i in range(len(timesteps)):
        if total_clusters[i] > 0:
            mean_size[i] = np.sum(hist_data[i] * size_bins) / total_clusters[i]
    
    # Read alpha data
    alpha_times, alpha_values = read_alpha_data(alpha_file_path)
    
    # Update font sizes
    plt.rcParams.update({
        'font.size': 30,
        'axes.titlesize': 34,
        'axes.labelsize': 32,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30,
    })
    
    # Create figure with specific gridspec to accommodate the colorbar
    # Create figure with specific gridspec to accommodate the colorbar
    if single_plot:
        # Generate just one comprehensive plot instead of animation
        t1 = len(timesteps) - 1  # Use the last timestep for the single plot
        plot_range = [t1]
    else:
        # Generate animation plots for each timestep
        plot_range = range(len(timesteps))
    
    for t1 in plot_range:
        #t1=1456
        fig = plt.figure(figsize=(35, 17))
        gs = fig.add_gridspec(1, 2, width_ratios=[20, 1], wspace=0.1)
        
        # Create main axis and colorbar axis
        ax_main = fig.add_subplot(gs[0, 0])
        ax_cbar = fig.add_subplot(gs[0, 1])
        
        # Create additional y-axes
        ax1 = ax_main  # Main axis for cluster size
        ax2 = ax1.twinx()  # For total clusters
        ax3 = ax1.twinx()  # For alpha values
        
        # Move ax2 and ax3 to the left side
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.tick_left()
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.tick_left()
        
    
        
        # Move the spines to the left side
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(True)
        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(True)
        
        # ... (scatter plot and line plotting code remains the same)
        
        # Set labels
        ax1.set_xlabel('Timestep')
        
    
        
        # Position axes with more space
        ax1.spines['left'].set_position(('outward', 0))     # Cluster Size (original left axis)
        ax2.spines['left'].set_position(('outward', 130))    # Total Clusters
        ax3.spines['left'].set_position(('outward', 240))   # Alpha
        
        # Create a special label for cluster size with two colors
        fig.canvas.draw()  # This is needed to get the transforms right
        
        # Get the position of the y-axis label
        ax1_ylabel_pos = ax1.yaxis.label.get_position()
        
        # Remove the original y-axis label
        ax1.set_ylabel('')
        
        # Add the two-part label
        ax1.text(-0.05, 0.5, 'Cluster Size                  , atoms', 
                 color='black', 
                 rotation=90, 
                 transform=ax1.transAxes,
                 verticalalignment='center')
        
        ax1.text(-0.05, 0.53, '/Mean size', 
                 color=(128/255, 0/255, 32/255), 
                 rotation=90, 
                 transform=ax1.transAxes,
                 verticalalignment='center')
       
        ax1.text(0.07, 0.62, f't={snapshots[t1]}', 
                 color='black', 
                 rotation=0, 
                 transform=ax1.transAxes,
                 horizontalalignment='left', 
                 fontsize=34,  # Adjust the size as needed
                 fontweight='bold')  # Set text to bold
       
        # Set other labels with adjusted positions
        ax2.set_ylabel('Total Number\nof Clusters', 
                      color='g', 
                      labelpad=25,
                      verticalalignment='center')
        
        ax3.set_ylabel('α', 
                      color='b', 
                      rotation=0, 
                      labelpad=35,
                      verticalalignment='center')
        
        # Adjust tick parameters to prevent overlap
        #ax1.tick_params(axis='y', labelcolor='k', pad=8)
       # ax2.tick_params(axis='y', labelcolor='g', pad=8)
       # ax3.tick_params(axis='y', labelcolor=(34/255, 139/255, 34/255), pad=8)
        
        # Optional: Adjust the y-axis tick label formats if needed
        # For example, using scientific notation or adjusting decimal places
        ax1.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        ax2.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
        
        ax2.set_ylabel('Total Number of Clusters',color=(147/255, 0/255, 147/255), labelpad=15)
        ax3.set_ylabel('α', color=(34/255, 139/255, 34/255), rotation=0, labelpad=25)
        
        # Set tick colors and adjust padding
        ax1.tick_params(axis='y', labelcolor='k', pad=5,labelsize=30)
        ax2.tick_params(axis='y', labelcolor=(147/255, 0/255, 147/255), pad=5,labelsize=30)
        ax3.tick_params(axis='y', labelcolor=(34/255, 139/255, 34/255), pad=5,labelsize=30)
        
        # Create meshgrid and prepare scatter plot data
        T, S = np.meshgrid(timesteps, size_bins)
        t_flat = T.flatten()
        s_flat = S.flatten()
        counts_flat = hist_data.T.flatten()
        
        mask = counts_flat > 0
        t_plot = t_flat[mask]
        s_plot = s_flat[mask]
        counts_plot = counts_flat[mask]
        # Find the appropriate indices based on timestep values
        # Find the index where timesteps start (skip very small timesteps)
        start_idx = 0
        for i, t in enumerate(timesteps):
            if t >= 10:  # Start from timestep 10 or higher
                start_idx = i
                break
        
        # Find the index for the current timestep t1
        if t1 < len(timesteps):
            t2 = t1
        else:
            t2 = len(timesteps) - 1
        # Calculate point sizes
        if log_scale:
            sizes = np.interp(np.log10(counts_plot), 
                             (np.log10(counts_plot.min()), np.log10(counts_plot.max())), 
                             (min_point_size, max_point_size))
            norm = LogNorm(vmin=counts_plot.min(), vmax=counts_plot.max())
        else:
            sizes = np.interp(counts_plot, 
                             (counts_plot.min(), counts_plot.max()), 
                             (min_point_size, max_point_size))
            norm = plt.Normalize(vmin=counts_plot.min(), vmax=counts_plot.max())
        
        # Plot scatter and mean size on first y-axis
        scatter = ax1.scatter(t_plot[t_plot < snapshots[t1]], s_plot[t_plot < snapshots[t1]],
                              c=counts_plot[t_plot < snapshots[t1]],
                              s=sizes[t_plot < snapshots[t1]],
                              alpha=alpha,  # Solid points for timestep < t1
                              norm=norm,
                              cmap='viridis',
                              label='Cluster Distribution')
        
        scatter_transparent = ax1.scatter(t_plot[t_plot >= snapshots[t1]], s_plot[t_plot >= snapshots[t1]],
                                           c=counts_plot[t_plot >= snapshots[t1]],
                                           s=sizes[t_plot >= snapshots[t1]],
                                           alpha=0.3,  # More visible semitransparent points for timestep >= t1
                                           norm=norm,
                                           cmap='viridis'
                                           )
        
        line1 = ax1.plot(timesteps[start_idx:t2], mean_size[start_idx:t2],  color=(128/255, 0/255, 32/255), 
                         linewidth=6, label='Mean Size')
        line11 = ax1.plot(timesteps[t2:], mean_size[t2:],  color=(128/255, 0/255, 32/255), 
                         linewidth=6,alpha=0.6)
        
        # Plot total clusters on second y-axis
        line2 = ax2.plot(timesteps[start_idx:t2], total_clusters[start_idx:t2], color=(147/255, 0/255, 147/255), 
                         linewidth=6, label='Total Number of Clusters')
        line21 = ax2.plot(timesteps[t2:], total_clusters[t2:], color=(147/255, 0/255, 147/255), 
                         linewidth=6,alpha=0.6)
        # Plot alpha values on third y-axis
        # Find appropriate indices for alpha data based on timestep values
        alpha_start_idx = 0
        for i, t in enumerate(alpha_times):
            if t >= 10:  # Start from timestep 10 or higher
                alpha_start_idx = i
                break
        
        # Find the index for alpha data corresponding to t2
        alpha_t2_idx = len(alpha_times) - 1
        for i, t in enumerate(alpha_times):
            if t >= timesteps[t2] if t2 < len(timesteps) else t >= timesteps[-1]:
                alpha_t2_idx = i
                break
        
        line3 = ax3.plot(alpha_times[alpha_start_idx:alpha_t2_idx], alpha_values[alpha_start_idx:alpha_t2_idx], color=(34/255, 139/255, 34/255), 
                         linewidth=6, label='α')
        line31 = ax3.plot(alpha_times[alpha_t2_idx:], alpha_values[alpha_t2_idx:], color=(34/255, 139/255, 34/255), 
                         linewidth=6,alpha=0.6)
        
        # Set scales
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        #ax2.set_yscale('log')
        
        # Set limits
        ax1.set_xlim(left=10, right=max(timesteps))
        
        # Set labels
    
        
        # Set tick colors
       # ax2.tick_params(axis='y', labelcolor='g')
        #ax3.tick_params(axis='y', labelcolor='b')
        
        # Add colorbar with full height and integer ticks
        min_count = int(np.floor(counts_plot.min()))
        max_count = int(np.ceil(counts_plot.max()))
        
        if log_scale:
            # For logarithmic scale, create ticks at each order of magnitude
            log_min = int(np.floor(np.log10(min_count)))
            log_max = int(np.ceil(np.log10(max_count)))
            tick_locations = np.logspace(log_min, log_max, log_max - log_min + 1)
            tick_labels = [f'{int(x)}' for x in tick_locations]
        else:
            # For linear scale, create evenly spaced integer ticks
            num_ticks = 10  # Adjust this number for more or fewer ticks
            tick_locations = np.linspace(min_count, max_count, num_ticks)
            tick_labels = [f'{int(x)}' for x in tick_locations]
    
        cbar = plt.colorbar(scatter, cax=ax_cbar, label='Number of Clusters', alpha=1.0)
        if log_scale:
            cbar.ax.set_yscale('log')
        
        # Set the new tick locations and labels
        cbar.set_ticks(tick_locations)
        cbar.set_ticklabels(tick_labels)
        cbar.outline.set_alpha(1.0)
        cbar.solids.set_alpha(1.0)
        cbar.ax.tick_params(labelsize=28)
        cbar.ax.yaxis.label.set_size(30)
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, 
                  loc='upper left')
        
        # Add grid
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        
        plt.suptitle('Cluster Evolution Analysis', y=0.95)
        
        # Save the plot
        if single_plot:
            plt.savefig(f'{output_prefix}cluster_distribution_analysis.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f'{output_prefix}combined_analysis_{snapshots[t1]}.png', dpi=300, bbox_inches='tight')
        
        # Close the figure to free memory
        plt.close(fig)
        return mean_size
    
    return mean_size

    