"""Command-line tests for local-density priors and diagnostic maps."""

import os
from pathlib import Path
import subprocess
import sys

import astropy.io.fits as pyfits
import numpy
import pytest

from nwaylib.fastskymatch import array2fits


def write_catalogue(directory, name, ra, dec):
	table = numpy.empty(len(ra), dtype=[('ID', 'i'), ('ra', 'f'), ('dec', 'f')])
	table['ID'] = numpy.arange(len(table))
	table['ra'] = ra
	table['dec'] = dec
	with array2fits(table, name) as hdulist:
		hdulist[1].header['SKYAREA'] = 1.0
		hdulist.writeto(directory / (name + '.fits'))


@pytest.fixture
def catalogues(tmp_path):
	write_catalogue(tmp_path, 'PRIMARY', [10, 10.1], 0)
	# The first two secondary sources have the same 6 arcsec separation
	# from their primary. The rest affect its KDE but are outside the match radius.
	ra = numpy.r_[10, 10.1, 10 + numpy.linspace(0.004, 0.008, 11)]
	write_catalogue(tmp_path, 'SECONDARY', ra, 6 / 3600)
	write_catalogue(tmp_path, 'THIRD', ra, 7 / 3600)
	return tmp_path


def run_nway(directory, output, *flags, third_catalogue=False):
	script = Path(__file__).resolve().parents[1] / 'nway.py'
	command = [sys.executable, '-B', str(script), 'PRIMARY.fits', '1', 'SECONDARY.fits', '1']
	if third_catalogue:
		command.extend(['THIRD.fits', '1'])
	command.extend(['--out', output, '--radius', '10', *flags])
	(directory / 'I_will_check_for_NWAY_updates_myself_thank_you').touch()
	env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(directory))
	subprocess.run(command, cwd=directory, env=env, check=True)
	with pyfits.open(directory / output) as hdulist:
		return hdulist[1].data.copy()


def test_local_density_options(catalogues):
	uniform = run_nway(catalogues, 'uniform.fits')
	local = run_nway(catalogues, 'local.fits', '--local-density')
	plot = run_nway(catalogues, 'plot.fits', '--plot-local-density')
	both = run_nway(catalogues, 'both.fits', '--local-density', '--plot-local-density')

	for table in (uniform, local):
		candidates = table[table['ncat'] == 2]
		candidates = candidates[numpy.argsort(candidates['PRIMARY_ID'])]
		numpy.testing.assert_array_equal(candidates['PRIMARY_ID'], [0, 1])
		numpy.testing.assert_allclose(candidates['dist_bayesfactor'],
			candidates['dist_bayesfactor'][0])
		if table is uniform:
			numpy.testing.assert_allclose(candidates['dist_post'], candidates['dist_post'][0])
		else:
			assert candidates['dist_post'][0] < candidates['dist_post'][1]

	for column in ('dist_post', 'p_any', 'p_i'):
		numpy.testing.assert_array_equal(plot[column], uniform[column])
		numpy.testing.assert_array_equal(both[column], local[column])

	assert {path.name for path in catalogues.glob('*.pdf')} == {
		'plot_SECONDARY.pdf', 'both_SECONDARY.pdf'}
	for name in ('plot_SECONDARY.pdf', 'both_SECONDARY.pdf'):
		assert (catalogues / name).read_bytes().startswith(b'%PDF')


def test_local_density_unrelated_association_correction(catalogues):
	uniform = run_nway(catalogues, 'uniform.fits', third_catalogue=True)
	local = run_nway(catalogues, 'local.fits', '--local-density', third_catalogue=True)

	for table in (uniform, local):
		primary_only = table[table['ncat'] == 1]
		assert len(primary_only) == 2
		assert numpy.all(primary_only['dist_bayesfactor_corrected'] > primary_only['dist_bayesfactor'])

	assert numpy.any(local['dist_bayesfactor_corrected'] != uniform['dist_bayesfactor_corrected'])
