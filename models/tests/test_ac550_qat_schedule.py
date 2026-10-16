from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class Ac550QatScheduleTests(unittest.TestCase):
    @staticmethod
    def _workspace_tempdir() -> tempfile.TemporaryDirectory:
        """Use the repository run directory; the sandboxed user temp folder is read-only."""
        project_root = Path(__file__).resolve().parents[2]
        runs_dir = project_root / "runs"
        runs_dir.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=runs_dir)

    def test_default_schedule_has_collect_weight_train_and_freeze_ranges(self) -> None:
        """The longer QAT recipe must stage weights before activation quantization."""
        from models.train.train_yolov5n_relu_ac550_qat import qat_phase_for_epoch

        arguments = {"observer_epochs": 5, "activation_quant_epoch": 15, "freeze_observer_epoch": 45}
        self.assertEqual(qat_phase_for_epoch(0, **arguments), "collect")
        self.assertEqual(qat_phase_for_epoch(4, **arguments), "collect")
        self.assertEqual(qat_phase_for_epoch(5, **arguments), "weight")
        self.assertEqual(qat_phase_for_epoch(14, **arguments), "weight")
        self.assertEqual(qat_phase_for_epoch(15, **arguments), "train")
        self.assertEqual(qat_phase_for_epoch(44, **arguments), "train")
        self.assertEqual(qat_phase_for_epoch(45, **arguments), "freeze")
        self.assertEqual(qat_phase_for_epoch(79, **arguments), "freeze")

    def test_batch_norm_freezes_from_declared_epoch(self) -> None:
        """BN statistics remain trainable until the scheduled freeze boundary."""
        from models.train.train_yolov5n_relu_ac550_qat import should_freeze_batch_norm

        self.assertFalse(should_freeze_batch_norm(19, freeze_bn_epoch=20))
        self.assertTrue(should_freeze_batch_norm(20, freeze_bn_epoch=20))

    def test_deployment_candidate_waits_for_frozen_qat_and_batch_norm(self) -> None:
        """A collection-only checkpoint must never be selected for FPGA export."""
        from models.train.train_yolov5n_relu_ac550_qat import is_deployment_candidate_epoch

        self.assertFalse(
            is_deployment_candidate_epoch(19, freeze_observer_epoch=15, freeze_bn_epoch=20)
        )
        self.assertTrue(
            is_deployment_candidate_epoch(20, freeze_observer_epoch=15, freeze_bn_epoch=20)
        )

    def test_final_checkpoint_is_copied_only_after_training_has_finished(self) -> None:
        """The epoch-20 freeze boundary must not write files from inside the training loop."""
        from models.train.train_yolov5n_relu_ac550_qat import copy_final_qat_checkpoint

        with self._workspace_tempdir() as temporary:
            save_dir = Path(temporary)
            weights = save_dir / "weights"
            weights.mkdir()
            (weights / "last.pt").write_bytes(b"frozen-qat-checkpoint")

            destination = copy_final_qat_checkpoint(
                save_dir,
                completed_epoch=39,
                deployment_candidate_start_epoch=20,
            )

            self.assertEqual(destination, weights / "best_qat.pt")
            self.assertEqual(destination.read_bytes(), b"frozen-qat-checkpoint")

    def test_final_checkpoint_can_publish_the_best_frozen_epoch_snapshot(self) -> None:
        """Deployment must use the best frozen epoch, not an earlier FP32 best or final epoch by default."""
        from models.train.train_yolov5n_relu_ac550_qat import copy_final_qat_checkpoint

        with self._workspace_tempdir() as temporary:
            save_dir = Path(temporary)
            weights = save_dir / "weights"
            weights.mkdir()
            (weights / "epoch51.pt").write_bytes(b"best-frozen-qat")

            destination = copy_final_qat_checkpoint(
                save_dir,
                completed_epoch=79,
                deployment_candidate_start_epoch=50,
                source_name="epoch51.pt",
            )

            self.assertEqual(destination.read_bytes(), b"best-frozen-qat")

    def test_fitness_callback_accepts_yolov5_single_value_array(self) -> None:
        """YOLOv5 emits fitness as a one-element NumPy array, not a Python float."""
        import numpy as np

        from models.train.train_yolov5n_relu_ac550_qat import scalar_fitness

        self.assertAlmostEqual(scalar_fitness(np.array([0.6125], dtype=np.float32)), 0.6125, places=6)

    def test_final_checkpoint_rejects_an_unfrozen_run(self) -> None:
        """An interrupted run before BN freezing must never become an export candidate."""
        from models.train.train_yolov5n_relu_ac550_qat import copy_final_qat_checkpoint

        with self._workspace_tempdir() as temporary:
            save_dir = Path(temporary)
            weights = save_dir / "weights"
            weights.mkdir()
            (weights / "last.pt").write_bytes(b"not-ready")

            with self.assertRaises(RuntimeError):
                copy_final_qat_checkpoint(
                    save_dir,
                    completed_epoch=19,
                    deployment_candidate_start_epoch=20,
                )

    def test_schedule_rejects_inconsistent_boundaries(self) -> None:
        """A frozen observer cannot precede the collection interval."""
        from models.train.train_yolov5n_relu_ac550_qat import validate_qat_schedule

        with self.assertRaises(ValueError):
            validate_qat_schedule(
                observer_epochs=5,
                activation_quant_epoch=4,
                freeze_observer_epoch=45,
                freeze_bn_epoch=50,
            )

    def test_vendor_training_import_exposes_both_yolo_and_qat_modules(self) -> None:
        """The entry point must work when imported from the project ``models`` namespace."""
        from models.train.train_yolov5n_relu_ac550_qat import _import_vendor_training

        vendor_train, _, prepare_ac550_qat, _, _ = _import_vendor_training()

        self.assertTrue(vendor_train.__file__.endswith("third_party\\yolov5-relu\\train.py"))
        self.assertTrue(callable(prepare_ac550_qat))

    def test_vendor_models_package_can_import_project_qat_checkpoint_class(self) -> None:
        """A saved QAT checkpoint is loadable by a fresh exporter Python process."""
        project_root = Path(__file__).resolve().parents[2]
        relu_root = project_root / "third_party" / "yolov5-relu"
        command = (
            "import sys; "
            f"sys.path.insert(0, r'{relu_root}'); "
            "import models.qat.ac550_qat_graph; "
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", command],
            cwd=relu_root,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
