"""Spherical density estimators and diagnostic maps."""
from __future__ import division, print_function

import os
import re

import matplotlib.pyplot as plt
import numpy
from astropy.coordinates import SkyCoord
from matplotlib.colors import LogNorm
from scipy.spatial import cKDTree
from scipy.special import logsumexp

def _valid_coordinates(lon, lat):
	return numpy.isfinite(lon) & numpy.isfinite(lat) & (numpy.abs(lat) <= 90)


def _log_kernel_norm(kappa):
	return numpy.log(kappa) - numpy.log(2 * numpy.pi * -numpy.expm1(-2 * kappa))

def unit_vectors(longitude, latitude):
	"""Convert longitude and latitude in degrees to Cartesian unit vectors."""
	lon = numpy.deg2rad(numpy.asarray(longitude, dtype=float))
	lat = numpy.deg2rad(numpy.asarray(latitude, dtype=float))
	cos_lat = numpy.cos(lat)
	return numpy.column_stack((cos_lat * numpy.cos(lon),
		cos_lat * numpy.sin(lon), numpy.sin(lat)))

def fit_local_density_model(sample_lon, sample_lat,
		neighbor_k=10, scale=1.0, min_bandwidth_deg=0.01,
		max_bandwidth_deg=10.0):
	"""
	Fit a spherical KDE to one-dimensional arrays of Galactic coordinates in degrees.
	Each bandwidth is the angular distance to the neighbor_k-th other source,
	multiplied by scale and clipped to the given bounds in degrees.
	"""
	sample_lon = numpy.asarray(sample_lon, dtype=float)
	sample_lat = numpy.asarray(sample_lat, dtype=float)
	if sample_lon.ndim != 1 or sample_lon.shape != sample_lat.shape:
		raise ValueError('Sample coordinates must be one-dimensional arrays of the same shape.')
	if not isinstance(neighbor_k, (int, numpy.integer)) or neighbor_k < 1:
		raise ValueError('neighbor_k must be a positive integer.')
	if not numpy.isfinite(scale) or scale <= 0:
		raise ValueError('scale must be finite and positive.')
	if (not numpy.isfinite(min_bandwidth_deg) or not numpy.isfinite(max_bandwidth_deg)
			or not 0 < min_bandwidth_deg <= max_bandwidth_deg):
		raise ValueError('Invalid minimum or maximum bandwidth.')

	valid = _valid_coordinates(sample_lon, sample_lat)
	lon = sample_lon[valid] % 360
	lat = sample_lat[valid]
	n_sample = len(lon)
	if n_sample <= neighbor_k:
		raise ValueError('The number of valid sources must be greater than neighbor_k.')

	vectors = unit_vectors(lon, lat)
	tree = cKDTree(vectors)
	# Chord distance preserves angular order. The extra neighbour
	# accounts for the source itself at distance zero.
	chords, _ = tree.query(vectors, k=[neighbor_k + 1])
	angles = 2 * numpy.arcsin(numpy.clip(chords[:, 0] / 2, 0, 1))
	bandwidths = numpy.clip(
		scale * numpy.rad2deg(angles), min_bandwidth_deg, max_bandwidth_deg)

	return dict(
		sample_lon=lon,
		sample_lat=lat,
		coordinates=SkyCoord(lon, lat, unit='deg', frame='galactic'),
		bandwidths_deg=bandwidths,
		neighbor_k=neighbor_k,
		scale=scale,
	)


def estimate_local_density(model, query_lon, query_lat):
	"""
	Evaluate the KDE at Galactic coordinates in degrees, preserving the query shape.
	The result is the density per steradian multiplied by the full-sky solid
	angle (4 pi). Each source contributes one unit of probability mass over
	the sphere.

	Invalid coordinates return NaN.
	"""
	query_lon = numpy.asarray(query_lon, dtype=float)
	query_lat = numpy.asarray(query_lat, dtype=float)
	if query_lon.shape != query_lat.shape:
		raise ValueError('query_lon and query_lat must have the same shape.')

	valid = _valid_coordinates(query_lon, query_lat)
	density = numpy.full(query_lon.shape, numpy.nan)
	if not numpy.any(valid):
		return density

	queries = SkyCoord(query_lon[valid] % 360, query_lat[valid], unit='deg', frame='galactic')
	sources = model['coordinates']
	n_sample = len(sources)
	if n_sample == 0:
		raise ValueError('The model contains no sample sources.')
	bandwidths = numpy.asarray(model['bandwidths_deg'], dtype=float)
	if (bandwidths.shape != (n_sample,) or not numpy.all(numpy.isfinite(bandwidths))
			or numpy.any(bandwidths <= 0)):
		raise ValueError('Invalid kernel bandwidths.')

	kappa = 1.0 / numpy.deg2rad(bandwidths)**2
	log_norm = _log_kernel_norm(kappa)
	theta = queries[:, None].separation(sources[None, :]).rad
	# cos(theta) - 1 loses precision for small angular separations.
	log_kernels = log_norm - 2 * kappa * numpy.sin(theta / 2)**2
	per_sr = numpy.exp(logsumexp(log_kernels, axis=1))

	density[valid] = 4 * numpy.pi * per_sr
	return density


def write_local_density_diagnostic(models, names, outfile):
	"""Save one Mollweide spherical-KDE map per counterpart catalogue."""
	output_root = os.path.splitext(outfile)[0]
	output_files = []
	# Display Galactic longitude increasing to the left.
	lon_grid = numpy.linspace(-180, 180, 361)
	lat_grid = numpy.linspace(-90, 90, 241)
	lon_mesh, lat_mesh = numpy.meshgrid(lon_grid, lat_grid)
	for model, name in zip(models, names):
		density_map = estimate_local_density(model, -lon_mesh, lat_mesh)
		valid = numpy.isfinite(density_map) & (density_map > 0)
		if not numpy.any(valid):
			raise ValueError('The local-density diagnostic has no positive values for "%s".' % name)
		density_plot = numpy.ma.masked_where(~valid, density_map)
		positive_density = density_map[valid]
		vmin = max(float(positive_density.min()), float(positive_density.max()) * 1e-8)
		vmax = float(positive_density.max())
		if vmax <= vmin:
			vmin = vmax * (1 - 1e-8)

		fig, ax = plt.subplots(
			figsize=(12, 6),
			subplot_kw={'projection': 'mollweide'})
		try:
			mesh = ax.pcolormesh(
				numpy.deg2rad(lon_mesh),
				numpy.deg2rad(lat_mesh),
				density_plot,
				shading='auto',
				cmap='viridis',
				norm=LogNorm(vmin=vmin, vmax=vmax),
				rasterized=True)
			if len(model['sample_lon']) > 20000:
				step = int(numpy.ceil(len(model['sample_lon']) / 20000))
			else:
				step = 1
			sample_lon = (model['sample_lon'] + 180) % 360 - 180
			ax.scatter(
				numpy.deg2rad(-sample_lon[::step]),
				numpy.deg2rad(model['sample_lat'][::step]),
				s=5,
				color='red',
				alpha=0.5,
				label='Counterpart sources')
			ax.grid(True)
			ax.set_xticks(numpy.deg2rad([-180, -90, 0, 90, 180]))
			ax.set_xticklabels(['180°', '90°', '0°', '270°', '180°'])
			ax.set_xlabel('GLON[°]')
			ax.set_ylabel('GLAT[°]')
			fig.colorbar(mesh, ax=ax, label='Full-sky-equivalent density')
			fig.tight_layout()
			safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('._') or 'catalogue'
			map_file = '%s_%s.pdf' % (output_root, safe_name)
			fig.savefig(map_file, format='pdf', bbox_inches='tight')
		finally:
			plt.close(fig)
		output_files.append(map_file)
	return output_files
