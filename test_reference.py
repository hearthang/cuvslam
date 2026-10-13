"""Numerical checks for the new optimization and evaluation mathematics."""
import unittest
import numpy as np
import gtsam
from gtsam.symbol_shorthand import X
from scipy.spatial.transform import Rotation

import reference
from evaluate_reference import score
from translation_refine import solve_positions


class ReferenceMathTests(unittest.TestCase):
    def test_measured_nonzero_return_is_not_forced_to_origin(self):
        t = np.linspace(0, 10, 101)
        truth = np.column_stack([np.sin(2*np.pi*t/10)+.025*t, t*0, t*0])
        drifted = truth + np.column_stack([.03*t, t*0, t*0])
        R = np.tile(np.eye(3), (len(t), 1, 1))
        out, _ = solve_positions(t, drifted, R, np.diff(drifted, axis=0), .01,
                                 np.array([[0, 100]]), np.array([[.25, 0, 0]]),
                                 loop_sigma=.001, loop_scale=100, acceleration_sigma=1)
        self.assertAlmostEqual(out[-1, 0], .25, places=3)
        self.assertGreater(out[:, 0].max(), .8)

    def test_sparse_smoothing_preserves_an_open_constant_velocity_path(self):
        t = np.linspace(0, 10, 101)
        p = np.column_stack([t, t*.2, t*0])
        R = np.tile(np.eye(3), (len(t), 1, 1))
        out, _ = solve_positions(t, p, R, np.diff(p, axis=0), .01,
                                 np.empty((0, 2), int), np.empty((0, 3)))
        np.testing.assert_allclose(out, p, atol=1e-7)
        self.assertGreater(np.linalg.norm(out[-1]-out[0]), 10)

    def test_sparse_smoothing_reduces_noise_without_shortening_motion(self):
        t = np.linspace(0, 10, 161)
        p = np.column_stack([t, np.zeros((len(t), 2))])
        rng = np.random.default_rng(3)
        noisy = p + rng.normal(0, .004, p.shape)
        noisy[0] = p[0]
        R = np.tile(np.eye(3), (len(t), 1, 1))
        out, _ = solve_positions(t, noisy, R, np.diff(noisy, axis=0), .01,
                                 np.empty((0, 2), int), np.empty((0, 3)), acceleration_sigma=.5)
        self.assertLess(np.linalg.norm(np.diff(out, n=2, axis=0)),
                        .4*np.linalg.norm(np.diff(noisy, n=2, axis=0)))
        self.assertLess(np.linalg.norm(out[-1]-p[-1]), .03)

    def test_acceleration_jacobian_with_unequal_intervals_and_rotated_frames(self):
        keys = [X(i) for i in range(3)]
        positions = [[0, 0, 0], [1, 0, 0], [3, 0, 0]]
        poses = [gtsam.Pose3(gtsam.Rot3(Rotation.from_rotvec([.2*i, -.1*i, .3*i]).as_matrix()), np.array(p, float))
                 for i, p in enumerate(positions)]
        values = gtsam.Values()
        for key, pose in zip(keys, poses):
            values.insert(key, pose)
        factor = reference.acceleration_factor(keys, .1, .2, 1)
        linear = factor.linearize(values)
        analytic = linear.getA().copy()
        numeric = np.zeros((3, 18))
        for i, key in enumerate(keys):
            for j in range(6):
                delta = np.zeros(6); delta[j] = 1e-6
                plus, minus = gtsam.Values(values), gtsam.Values(values)
                plus.update(key, poses[i].retract(delta))
                minus.update(key, poses[i].retract(-delta))
                numeric[:, i*6+j] = (factor.unwhitenedError(plus)-factor.unwhitenedError(minus)) / 2e-6
        np.testing.assert_allclose(analytic, numeric, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(factor.unwhitenedError(values), 0, atol=1e-12)

    def test_gyro_comparison_detects_wrong_direction_not_just_angle_magnitude(self):
        t = np.arange(5, dtype=float)
        r = Rotation.from_rotvec(np.outer(t, [.1, 0, 0]))
        a = np.column_stack([t, np.zeros((5, 3)), r.as_quat()])
        gyro = Rotation.from_rotvec(np.tile([-.1, 0, 0], (4, 1)))
        result = score(a, t, gyro, np.zeros(4), np.ones(4)*6)
        self.assertEqual(result['gyro_bad'], 4)
        self.assertAlmostEqual(result['gyro_p95_deg'], np.degrees(.2), places=6)

    def test_open_path_is_not_reported_as_closed(self):
        t = np.arange(5, dtype=float)
        a = np.column_stack([t, t, np.zeros((5, 2)), np.tile([0, 0, 0, 1], (5, 1))])
        gyro = Rotation.identity(4)
        result = score(a, t, gyro, np.ones(4), np.zeros(4))
        self.assertEqual(result['closure_m'], 4)
        self.assertEqual(result['path_m'], 4)
        self.assertEqual(result['acc_p95'], 0)


if __name__ == '__main__':
    unittest.main()
