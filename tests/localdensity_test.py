"""Tests for the adaptive spherical density estimator."""

import numpy
import pytest

from nwaylib.localdensity import estimate_local_density, fit_local_density_model


def test_fit_local_density_bandwidths():
	model = fit_local_density_model([359.8, 0.2, 3, 50, 65], numpy.zeros(5),
		neighbor_k=1, min_bandwidth_deg=0.01, max_bandwidth_deg=10)
	numpy.testing.assert_allclose(model['bandwidths_deg'], [0.4, 0.4, 2.8, 10, 10])


def test_estimate_local_density_normalization():
	model = fit_local_density_model([0, 0], [-90, 90], neighbor_k=1,
		min_bandwidth_deg=20, max_bandwidth_deg=20)
	# For kernels at the poles, the spherical mean is the integral over
	# sin(latitude) from -1 to 1, divided by two.
	sin_latitude, weights = numpy.polynomial.legendre.leggauss(80)
	latitude = numpy.rad2deg(numpy.arcsin(sin_latitude))
	density = estimate_local_density(model, numpy.zeros_like(latitude), latitude)
	numpy.testing.assert_allclose(numpy.dot(weights, density) / 2, 2, atol=1e-12)


def test_invalid_coordinates():
	model = fit_local_density_model([0, 1, numpy.nan, 2], [0, 0, 0, 91], neighbor_k=1)
	numpy.testing.assert_array_equal(model['sample_lon'], [0, 1])
	density = estimate_local_density(model, [0, 0], [0, 91])
	assert numpy.isfinite(density[0]) and density[0] > 0
	assert numpy.isnan(density[1])

	with pytest.raises(ValueError, match='greater than neighbor_k'):
		fit_local_density_model(numpy.arange(10), numpy.zeros(10))
