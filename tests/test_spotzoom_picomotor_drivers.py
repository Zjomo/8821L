from __future__ import annotations

import sys
import unittest
from unittest import mock

from SpotZoom import NewportMRC4MirrorStage, NewportPicomotorZAxis, parse_args


class FakePicoMotorController:
    def __init__(self, axes=(1, 2, 3, 4)) -> None:
        self._axes = list(axes)
        self.moves = []
        self.wait_calls = []
        self.stop_calls = []
        self.vel_calls = []
        self.closed = False

    def get_id(self):
        return "FAKE-8742"

    def axes(self):
        return list(self._axes)

    def require_axes(self, axes):
        requested = [int(axis) for axis in axes]
        missing = [axis for axis in requested if axis not in self._axes]
        if missing:
            raise RuntimeError(f"Missing axes: {missing}")
        return list(self._axes)

    def set_vel(self, axis=1, speed=None, accel=None):
        self.vel_calls.append((axis, speed, accel))

    def move_rel(self, axis=1, steps=0, wait=True):
        self.moves.append((axis, steps, wait))

    def wait(self, axis=1):
        self.wait_calls.append(axis)

    def stop(self, axis="all", immediate=False):
        self.stop_calls.append((axis, immediate))

    def close(self):
        self.closed = True


class SpotZoomPicomotorDriverTests(unittest.TestCase):
    def test_mrc_shared_mode_routes_virtual_xy_to_both_mirrors(self) -> None:
        fake = FakePicoMotorController()
        stage = NewportMRC4MirrorStage(
            controller=fake,
            wait_each_move=False,
            virtual_axis_mode="shared",
        )

        stage.move_x(12)
        stage.move_y(-7)
        stage.wait_startup_axis("x")
        stage.wait_startup_axis("y")

        self.assertEqual(
            fake.moves,
            [
                (1, 12, False),
                (3, 12, False),
                (2, -7, False),
                (4, -7, False),
            ],
        )
        self.assertEqual(fake.wait_calls, [1, 3, 2, 4])

    def test_mrc_explicit_mirror_moves_apply_per_axis_signs(self) -> None:
        fake = FakePicoMotorController()
        stage = NewportMRC4MirrorStage(
            controller=fake,
            mirror1_x_sign=-1,
            mirror1_y_sign=1,
            mirror2_x_sign=1,
            mirror2_y_sign=-1,
        )

        stage.move_mirrors(
            mirror1_x_steps=5,
            mirror1_y_steps=6,
            mirror2_x_steps=-7,
            mirror2_y_steps=8,
        )

        self.assertEqual(
            fake.moves,
            [
                (1, -5, True),
                (2, 6, True),
                (3, -7, True),
                (4, -8, True),
            ],
        )

    def test_picomotor_z_axis_quantizes_steps_and_applies_sign(self) -> None:
        fake = FakePicoMotorController(axes=(1, 2))
        z_axis = NewportPicomotorZAxis(
            controller=fake,
            axis=2,
            hw_sign=-1,
            wait_each_move=False,
        )

        z_axis.move_up(1.2)
        z_axis.move_down(2.6)
        z_axis.wait_startup_axis()
        z_axis.close()

        self.assertEqual(fake.moves, [(2, -1, False), (2, 3, False)])
        self.assertEqual(fake.wait_calls, [2])
        self.assertIn((2, False), fake.stop_calls)
        self.assertTrue(fake.closed)

    def test_parse_args_accepts_mrc4_and_picomotor_options(self) -> None:
        argv = [
            "SpotZoom.py",
            "--xy-driver",
            "newport-mrc4",
            "--mrc-virtual-axis-mode",
            "shared",
            "--z-driver",
            "picomotor",
            "--z-picomotor-conn",
            "1",
            "--z-picomotor-axis",
            "2",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.xy_driver, "newport-mrc4")
        self.assertEqual(args.mrc_virtual_axis_mode, "shared")
        self.assertEqual(args.z_driver, "picomotor")
        self.assertEqual(args.z_picomotor_conn, 1)
        self.assertEqual(args.z_picomotor_axis, 2)


if __name__ == "__main__":
    unittest.main()
