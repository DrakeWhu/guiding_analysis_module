"""Small synthetic rendering regression; no SUNRISE raws required."""
import contextlib
import csv
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


class SyntheticSeries:
    def __init__(self, path, **kwargs):
        self.iterations = np.array([0, 4000])
        self.t = np.array([0., 1e-12])

    def get_particle(self, **kwargs):
        time = kwargs['iteration']/4000
        arrays = {'x':np.array([-1,0,1])*1e-6*(1+time), 'y':np.zeros(3),
                  'z':np.array([-2,-1,0])*1e-6+time*1e-3,
                  'ux':np.array([-2,0,2.]), 'uy':np.zeros(3),
                  'uz':np.array([10,120,200.]), 'w':np.array([1,2,3.])*1e6}
        return [arrays[key] for key in kwargs['var_list']]

    def get_field(self, **kwargs):
        time = kwargs['iteration']/4000
        # Reverse axis order intentionally: the renderer must transpose.
        rho = -1.602176634e-19*1e24*np.full((5,4),1+time+.1*kwargs['theta'])
        info = SimpleNamespace(axes={0:'z',1:'r'},
                               z=np.linspace(-5e-6,0,5)+time*1e-3,
                               r=np.linspace(-3e-6,3e-6,4))
        return rho, info


class BeamVisualizationTests(unittest.TestCase):
    def test_complete_series_weighted_charge_density_axes_and_gifs(self):
        script = Path(__file__).resolve().parents[1]/'scripts/animate_beam_evolution.py'
        spec = importlib.util.spec_from_file_location('beam_visualization',script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root/'case'; case.mkdir()
            (case/'warpx_used_inputs').write_text('my_constants.n0 = 1e24\n')
            (case/'guiding_metrics.csv').write_text(
                'iteration,time_fs,z_max_um\n0,0,0\n4000,1000,1000\n')
            args = [str(script),'--case-dir',str(case),'--outdir',str(root/'out')]
            with patch.object(module,'OpenPMDTimeSeries',SyntheticSeries), \
                    patch.object(sys,'argv',args), contextlib.redirect_stdout(io.StringIO()):
                module.main()
            for name in ('phase_space_full.gif','rho_electrons_full.gif'):
                with Image.open(root/'out'/name) as image:
                    self.assertEqual(image.n_frames,2)
            for name in ('phase_frames','rho_frames'):
                self.assertEqual(len(list((root/'out'/name).glob('*.png'))),2)
            with (root/'out/frame_manifest.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertAlmostEqual(float(rows[0]['charge_forward_50MeV_pC']),
                                   5e6*module.QE*1e12)
            field,z,r = module.density(SyntheticSeries('fields'),0,0.,1e24,0.)
            self.assertEqual(field.shape,(4,5))
            np.testing.assert_allclose(field,1.)

    def test_multispecies_particles_and_density_sum(self):
        script = Path(__file__).resolve().parents[1]/'scripts/animate_beam_evolution.py'
        spec = importlib.util.spec_from_file_location('beam_visualization_multi', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        series = SyntheticSeries('unused')
        result = module.particles(series, 0, ['background', 'nitrogen'])
        self.assertEqual(len(result[0]), 6)
        self.assertAlmostEqual(result[6].sum(), 12e6)
        field, _, _ = module.density(series, 0, 0., 1e24, 0., ['rho_a','rho_b'])
        np.testing.assert_allclose(field, 2.)

    def test_density_grid_mismatch_is_rejected(self):
        script = Path(__file__).resolve().parents[1]/'scripts/animate_beam_evolution.py'
        spec = importlib.util.spec_from_file_location('beam_visualization_bad', script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        class BadSeries(SyntheticSeries):
            def get_field(self, **kwargs):
                rho, info = super().get_field(**kwargs)
                if kwargs['field'] == 'rho_b':
                    info.z = info.z + 1e-6
                return rho, info
        with self.assertRaisesRegex(ValueError, 'grids differ'):
            module.density(BadSeries('unused'), 0, 0., 1e24, 0., ['rho_a','rho_b'])


if __name__ == '__main__':
    unittest.main()
