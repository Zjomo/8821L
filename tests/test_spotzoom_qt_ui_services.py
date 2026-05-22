from __future__ import annotations

import unittest
from pathlib import Path

from spotzoom_qt_ui.models import RunMode
from spotzoom_qt_ui.services import DeviceRegistryService, ModuleCatalogService, RuntimeControlService


class SpotZoomQtUiServicesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parent.parent
        self.runtime = RuntimeControlService(repo_root=self.repo_root)

    def test_build_command_contains_startup_check_only(self) -> None:
        profile = self.runtime.default_profile()
        profile.detector_backend = "classic"
        command = self.runtime.build_command(profile, startup_check_only=True)
        self.assertIn("--startup-motion-check-only", command)
        self.assertIn("--detector-backend", command)

    def test_build_command_single_step_forces_max_iterations_1(self) -> None:
        profile = self.runtime.default_profile()
        profile.max_iterations = 8
        command = self.runtime.build_command(profile, single_step=True)
        idx = command.index("--max-iterations")
        self.assertEqual(command[idx + 1], "1")

    def test_mrc_allocation_preview_for_shared_mode(self) -> None:
        profile = self.runtime.default_profile()
        profile.xy_driver = "newport-mrc4"
        profile.mrc_virtual_axis_mode = "shared"
        profile.mrc_mirror1_x_axis = 1
        profile.mrc_mirror2_x_axis = 3
        profile.mrc_mirror1_y_axis = 2
        profile.mrc_mirror2_y_axis = 4
        reg = DeviceRegistryService()
        preview = reg.mrc_allocation_preview(profile)
        self.assertIn("virtual X", preview)
        self.assertIn("1", preview)
        self.assertIn("4", preview)

    def test_module_catalog_non_empty(self) -> None:
        catalog = ModuleCatalogService()
        rows = catalog.list_modules(self.runtime.default_profile())
        self.assertGreater(len(rows), 0)
        self.assertTrue(any(row.type_label for row in rows))

    def test_snapshot_mode_label_matches_run_mode(self) -> None:
        profile = self.runtime.default_profile()
        profile.run_mode = RunMode.REAL
        snapshot = self.runtime.build_snapshot(profile)
        self.assertEqual(snapshot.mode_label, "真实设备")


if __name__ == "__main__":
    unittest.main()
