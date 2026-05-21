from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from SpotZoom_Machine_Learning_Unified import (
    OptimizationType,
    build_module_adapter,
    get_module_spec,
    get_modules_by_type,
    get_modules_by_version,
    list_module_specs,
    run_system_pipeline,
)


class SpotZoomMLUnifiedTests(unittest.TestCase):
    def test_registry_discovers_all_versioned_modules(self) -> None:
        specs = list_module_specs()
        self.assertEqual(len(specs), 50)
        self.assertEqual(sorted({spec.version for spec in specs}), [2, 3, 4, 5, 6, 7])

    def test_grouping_and_adapter_instantiation(self) -> None:
        enhancement_modules = {spec.module_name for spec in get_modules_by_type(OptimizationType.IMAGE_ENHANCEMENT)}
        self.assertIn("self_supervised_denoiser", enhancement_modules)
        self.assertIn("deep_image_prior_enhancer", enhancement_modules)

        spec = get_module_spec(7, "topology_aware_optimizer")
        self.assertEqual(spec.primary_symbol, "TopologyAwareOptimizer")

        adapter = build_module_adapter(2, "closed_loop_ao_controller")
        config = adapter.build_config(history_length=8)
        instance = adapter.build_instance(config=config)
        self.assertEqual(config.history_length, 8)
        self.assertEqual(instance.__class__.__name__, "ClosedLoopAOController")

    def test_pipeline_generates_docs_and_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary = run_system_pipeline(Path(tmp_dir))
            output_dir = Path(summary.output_dir)
            self.assertTrue((output_dir / "INDEX.md").exists())
            self.assertTrue((output_dir / "catalog.json").exists())
            docs = sorted(output_dir.glob("v*_*.md"))
            self.assertEqual(len(docs), 50)
            self.assertEqual(summary.module_count, 50)
            self.assertEqual(summary.document_count, 51)


if __name__ == "__main__":
    unittest.main()
