# -*- coding: utf-8 -*-
"""
GUI-integrated cluster analysis for 3D CA simulations.

- analyze_xyz_file(...)
Internal helpers: create_3d_array, get_periodic_neighbors, find_clusters_and_labels, analyze_clusters, _generate_histogram_data
"""

import numpy as np
from networkx import Graph, connected_components
from collections import Counter
import json
from pathlib import Path
from typing import List, Tuple, Any, Dict
import XYZ_loader as loader


def create_3d_array(atom_list: List[Tuple[Any, Tuple[int, int, int], Any]], size: int) -> np.ndarray:
    """
    Create a 3D array from list of atoms with their coordinates and types.
    """
    # Initialize 3D array with zeros
    array_3d = np.zeros((size, size, size), dtype=str)
    
    # Fill array with atom types at their coordinates
    for (x, y, z), atom_type, _ in atom_list:
        array_3d[int(x), int(y), int(z)] = str(atom_type)
    
    return array_3d


def get_periodic_neighbors(x: int, y: int, z: int, size: int) -> List[Tuple[int, int, int]]:
    """
    Get all 26 neighbors considering periodic boundary conditions.
    """
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == dy == dz == 0:
                    continue
                
                # Apply periodic boundary conditions
                nx = (x + dx) % size
                ny = (y + dy) % size
                nz = (z + dz) % size
                
                neighbors.append((nx, ny, nz))
    
    return neighbors


def find_clusters_and_labels(array_3d: np.ndarray, needed_type: Any, min_cluster: int) -> Tuple[List[set], np.ndarray]:
    """
    Find all clusters of specified type considering periodic boundaries and create labeled array.
    Returns:
    - List of sets containing coordinates for each cluster
    - 3D array with cluster labels (0 for non-cluster elements)
    """
    size = array_3d.shape[0]
    G = Graph()
    needed_type = str(needed_type)
    
    coords = np.where(array_3d == needed_type)
    points = list(zip(coords[0], coords[1], coords[2]))
    G.add_nodes_from(points)
    
    for point in points:
        x, y, z = point
        neighbors = get_periodic_neighbors(x, y, z, size)
        for nx, ny, nz in neighbors:
            if array_3d[nx, ny, nz] == needed_type:
                G.add_edge((x, y, z), (nx, ny, nz))
    
    all_clusters = list(connected_components(G))
    
    clusters = [cluster for cluster in all_clusters if len(cluster) >= min_cluster]
    
    # Create labeled array
    labeled_array = np.zeros_like(array_3d, dtype=int)
    for cluster_idx, cluster in enumerate(clusters, start=1):
        for x, y, z in cluster:
            labeled_array[x, y, z] = cluster_idx
    
    return clusters, labeled_array


def analyze_clusters(clusters: List[set]) -> Tuple[int, float, Counter]:
    """
    Analyze clusters to get number of clusters, mean size, and size distribution.
    """
    # Analyze clusters
    if not clusters:
        return 0, 0, Counter()
    
    cluster_sizes = [len(cluster) for cluster in clusters]
    return (
        len(clusters),
        np.mean(cluster_sizes),
        Counter(cluster_sizes)
    )


def _generate_histogram_data(results: Dict[str, Any], output_file: Path) -> None:
    """
    Generate histogram data file in the format expected by the plotting functions.
    
    Args:
        results: Results dictionary from cluster analysis
        output_file: Path to save the histogram data file
    """
    if "snapshot_results" not in results:
        return
    
    # Collect all unique cluster sizes across all snapshots
    all_sizes = set()
    for snapshot in results["snapshot_results"]:
        if "size_distribution" in snapshot:
            all_sizes.update(snapshot["size_distribution"].keys())
    
    # Sort sizes for consistent ordering
    size_bins = sorted([int(size) for size in all_sizes])
    
    # Create histogram data matrix
    timesteps = []
    hist_data = []
    
    for snapshot in results["snapshot_results"]:
        timestep = snapshot.get("timestep", 0)
        timesteps.append(timestep)
        
        # Create histogram row for this timestep
        row = []
        size_dist = snapshot.get("size_distribution", {})
        
        for size in size_bins:
            count = size_dist.get(str(size), 0)
            row.append(count)
        
        hist_data.append(row)
    
    # Write histogram data file
    with open(output_file, 'w') as f:
        # Write header
        header = "Timestep, " + ", ".join([f"Size_{size}" for size in size_bins])
        f.write(header + "\n")
        
        # Write data rows
        for i, timestep in enumerate(timesteps):
            row_data = [str(timestep)] + [str(count) for count in hist_data[i]]
            f.write(", ".join(row_data) + "\n")
    
    # Generated histogram data for plotting


def analyze_xyz_file(xyz_file_path: str, output_dir: str, lattice_size: int, 
                    target_type: int = 2, min_cluster_size: int = 1, 
                    xyz_interval: int = 1, progress_callback=None,
                    timesteps_override: list[int] | None = None) -> Dict[str, Any]:
    """
    Analyze a single XYZ file for cluster properties and save results.
    
    Args:
        xyz_file_path: Path to the XYZ file to analyze
        output_dir: Output directory for results
        lattice_size: Size of the 3D lattice
        target_type: Atom type to analyze (default: 2 for crystalline atoms)
        min_cluster_size: Minimum cluster size to consider
        xyz_interval: XYZ snapshot interval steps (for timestep calculation)
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Dictionary with analysis results
    """
    # Starting cluster analysis
    
    if progress_callback:
        progress_callback()
    
    # Create cluster_analysis subdirectory for JSON files
    cluster_analysis_dir = Path(output_dir) / "cluster_analysis"
    cluster_analysis_dir.mkdir(parents=True, exist_ok=True)
    # Created cluster analysis directory
    
    if progress_callback:
        progress_callback()
    
    # Load XYZ file
    try:
        xyz_loader = loader.XYZLoader(xyz_file_path)
        # Loaded XYZ file successfully
        
        if progress_callback:
            progress_callback()
        
        # Load all snapshots
        snapshots = list(xyz_loader.load_snapshots(range(xyz_loader.get_n_snapshots())))
        
        if progress_callback:
            progress_callback()
        
    except Exception as e:
        print(f"Error loading XYZ file: {e}")
        return {"error": str(e)}
    
    # Analyze each snapshot
    results = []
    cluster_data = []
    
    for snapshot_idx, atom_list in enumerate(snapshots):
        print(f"Analyzing snapshot {snapshot_idx + 1}/{len(snapshots)}")
        
        if progress_callback:
            progress_callback()
        
        # Calculate timestep for this snapshot
        if timesteps_override is not None and snapshot_idx < len(timesteps_override):
            timestep = timesteps_override[snapshot_idx]
        else:
            timestep = snapshot_idx * xyz_interval
        
        # Create 3D array
        array_3d = create_3d_array(atom_list, lattice_size)
        
        # Find clusters
        clusters, labeled_array = find_clusters_and_labels(array_3d, target_type, min_cluster_size)
        
        # Analyze clusters
        n_clusters, mean_size, size_distribution = analyze_clusters(clusters)
        
        # Store results for this snapshot
        snapshot_result = {
            'snapshot': snapshot_idx,
            'timestep': timestep,
            'number_of_clusters': n_clusters,
            'mean_cluster_size': float(mean_size),
            'size_distribution': dict(size_distribution)
        }
        results.append(snapshot_result)
        
        # Store data for .dat file
        cluster_data.append({
            'timestep': timestep,
            'n_clusters': n_clusters,
            'mean_size': float(mean_size)
        })
        
        # Save individual snapshot JSON (optional, for detailed analysis)
        json_file = cluster_analysis_dir / f"snapshot_{snapshot_idx}_results.json"
        with open(json_file, 'w') as f:
            json.dump(snapshot_result, f, indent=4)
        
        if progress_callback:
            progress_callback()
    
    # Save comprehensive results
    comprehensive_results = {
        'analysis_parameters': {
            'xyz_file': xyz_file_path,
            'lattice_size': lattice_size,
            'target_type': target_type,
            'min_cluster_size': min_cluster_size,
            'xyz_interval': xyz_interval,
            'total_snapshots': len(snapshots)
        },
        'snapshot_results': results
    }
    
    # Save comprehensive JSON
    comprehensive_json = cluster_analysis_dir / "comprehensive_cluster_analysis.json"
    with open(comprehensive_json, 'w') as f:
        json.dump(comprehensive_results, f, indent=4)
    print(f"Saved comprehensive results: {comprehensive_json}")
    
    if progress_callback:
        progress_callback()
    
    # Save .dat file for time series analysis (in output directory, not subfolder)
    dat_file = Path(output_dir) / "cluster_analysis.dat"
    with open(dat_file, 'w') as f:
        f.write("# Timestep\tNumber_of_Clusters\tMean_Cluster_Size\n")
        for data in cluster_data:
            f.write(f"{data['timestep']}\t{data['n_clusters']}\t{data['mean_size']:.6f}\n")
    print(f"Saved cluster analysis data: {dat_file}")
    
    if progress_callback:
        progress_callback()
    
    # Generate histogram data for plotting
    print("Generating histogram data for plotting...")
    histogram_file = cluster_analysis_dir / "_size_distribution_data.txt"
    _generate_histogram_data(results, histogram_file)
    print(f"Saved histogram data: {histogram_file}")
    
    if progress_callback:
        progress_callback()
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total snapshots analyzed: {len(snapshots)}")
    if cluster_data:
        total_clusters = sum(data['n_clusters'] for data in cluster_data)
        avg_clusters = total_clusters / len(cluster_data)
        avg_mean_size = sum(data['mean_size'] for data in cluster_data) / len(cluster_data)
        print(f"Average number of clusters per snapshot: {avg_clusters:.2f}")
        print(f"Average mean cluster size: {avg_mean_size:.2f}")
    
    return comprehensive_results


 
