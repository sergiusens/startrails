"""Snap installs point NVIDIA users at the cuda component only while it is missing."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from startrails.snap import cuda_component_hint


class CudaComponentHintTests(TestCase):
    def setUp(self):
        directory = TemporaryDirectory(prefix="startrails_snap_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.pci = self.root / "pci"
        self.components = self.root / "components"
        self.pci.mkdir()
        self.components.mkdir()
        self.enterContext(patch.dict(os.environ, {"SNAP_COMPONENTS": str(self.components)}))

    def add_pci_device(self, name: str, vendor: str, device_class: str):
        device = self.pci / name
        device.mkdir()
        (device / "vendor").write_text(f"{vendor}\n")
        (device / "class").write_text(f"{device_class}\n")

    def test_nvidia_gpu_without_component_suggests_install(self):
        self.add_pci_device("0000:00:02.0", "0x8086", "0x030000")
        self.add_pci_device("0000:01:00.0", "0x10de", "0x030200")
        self.assertIn("snap install startrails+cuda", cuda_component_hint(self.pci))

    def test_installed_component_suppresses_hint(self):
        self.add_pci_device("0000:01:00.0", "0x10de", "0x030000")
        (self.components / "cuda").mkdir()
        self.assertEqual(cuda_component_hint(self.pci), "")

    def test_nvidia_non_display_device_is_ignored(self):
        # NVIDIA also ships audio and USB functions on the same card.
        self.add_pci_device("0000:01:00.1", "0x10de", "0x040300")
        self.assertEqual(cuda_component_hint(self.pci), "")
