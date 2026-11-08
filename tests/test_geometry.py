"""Check pure geometry helpers without loading ROS2 or opening the serial port."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / 'main_picker.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
rotation = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'quaternion_to_rotation_matrix')
picker = next(n for n in tree.body if isinstance(n, ast.ClassDef)
              and n.name == 'CucumberPickerNode')
depth = next(n for n in picker.body if isinstance(n, ast.FunctionDef)
             and n.name == 'get_depth_mm')
namespace = {'np': np}
exec(compile(ast.Module(body=[rotation, depth], type_ignores=[]),
             str(SOURCE), 'exec'), namespace)


class GeometryTests(unittest.TestCase):
    def test_identity_rotation(self):
        q = SimpleNamespace(x=0, y=0, z=0, w=1)
        np.testing.assert_allclose(namespace['quaternion_to_rotation_matrix'](q),
                                   np.eye(3))

    def test_quarter_turn_about_z(self):
        q = SimpleNamespace(x=0, y=0, z=np.sqrt(0.5), w=np.sqrt(0.5))
        matrix = namespace['quaternion_to_rotation_matrix'](q)
        np.testing.assert_allclose(matrix @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(matrix), 1)

    def test_depth_ignores_invalid_samples(self):
        image = np.array([[0, 100, 900], [200, 300, 0], [0, 400, 0]])
        self.assertEqual(namespace['get_depth_mm'](
            SimpleNamespace(depth_roi_size=3), image, 1, 1), 300)

    def test_depth_at_image_edge(self):
        image = np.array([[100, 200], [300, 400]])
        self.assertEqual(namespace['get_depth_mm'](
            SimpleNamespace(depth_roi_size=3), image, 0, 0), 250)

    def test_empty_depth(self):
        self.assertEqual(namespace['get_depth_mm'](
            SimpleNamespace(depth_roi_size=3), np.zeros((3, 3)), 1, 1), 0)


if __name__ == '__main__':
    unittest.main()
