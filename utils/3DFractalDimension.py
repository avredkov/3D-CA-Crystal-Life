# -*- coding: utf-8 -*-
"""
Minimal fractal dimension calculation utilities.

Exposes a single function `fractal_dimension` suitable for reuse.
"""

import numpy as np
import matplotlib.pyplot as plt


def fractal_dimension(array, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):
	"""Calculate the fractal dimension of a 3D numpy array using box counting.

	Args:
		array (np.ndarray): 3D lattice array.
		max_box_size (int | None): Largest power-of-two box exponent; inferred from array if None.
		min_box_size (int): Smallest power-of-two box exponent.
		n_samples (int): Number of scales to evaluate.
		n_offsets (int): Number of offsets to minimize N(s) per scale.
		plot (bool): If True, render the log-log fit.

	Returns:
		float: Estimated fractal dimension.
	"""
	if max_box_size is None:
		max_box_size = int(np.floor(np.log2(np.min(array.shape))))
	scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
	scales = np.unique(scales)

	locs = np.where(array == 2)
	voxels = np.array([(x, y, z) for x, y, z in zip(*locs)])

	Ns = []
	for scale in scales:
		touched = []
		offsets = [0] if n_offsets == 0 else np.linspace(0, scale, n_offsets)
		for offset in offsets:
			bin_edges = [np.arange(0, i, scale) for i in array.shape]
			bin_edges = [np.hstack([0 - offset, x + offset]) for x in bin_edges]
			H1, _ = np.histogramdd(voxels, bins=bin_edges)
			touched.append(np.sum(H1 > 0))
		Ns.append(touched)
	Ns = np.array(Ns).min(axis=1)

	# Keep scales at which Ns changes and positive
	scales = np.array([np.min(scales[Ns == x]) for x in np.unique(Ns)])
	Ns = np.unique(Ns)
	Ns = Ns[Ns > 0]
	scales = scales[:len(Ns)]

	coeffs = np.polyfit(np.log(1 / scales), np.log(Ns), 1)

	if plot:
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.scatter(np.log(1 / scales), np.log(np.unique(Ns)), c="teal", label="Measured ratios")
		ax.set_ylabel("log N(ε)")
		ax.set_xlabel("log 1/ε")
		fitted_y_vals = np.polyval(coeffs, np.log(1 / scales))
		ax.plot(np.log(1 / scales), fitted_y_vals, "k--", label=f"Fit: {np.round(coeffs[0], 3)}X+{coeffs[1]}")
		ax.legend()
	return coeffs[0]

    
