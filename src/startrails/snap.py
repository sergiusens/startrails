"""Snap package integration: snapd delivers updates and the optional cuda component."""

import os
from pathlib import Path

SNAP_NAME = "startrails"
CUDA_COMPONENT = "cuda"
NVIDIA_PCI_VENDOR = "0x10de"
# PCI base class 0x03 covers VGA, 3D, and other display controllers.
PCI_DISPLAY_CLASS_PREFIX = "0x03"
PCI_DEVICES_DIR = Path("/sys/bus/pci/devices")


def is_snap() -> bool:
    """Whether this process runs from the StarTrails snap rather than another snap's environment."""
    return os.environ.get("SNAP_NAME") == SNAP_NAME


def has_nvidia_gpu(pci_devices: Path = PCI_DEVICES_DIR) -> bool:
    """Look for an NVIDIA display controller, whether or not its driver is loaded."""
    try:
        devices = list(pci_devices.iterdir())
    except OSError:
        return False
    for device in devices:
        try:
            vendor = (device / "vendor").read_text().strip()
            device_class = (device / "class").read_text().strip()
        except OSError:
            continue
        if vendor == NVIDIA_PCI_VENDOR and device_class.startswith(PCI_DISPLAY_CLASS_PREFIX):
            return True
    return False


def cuda_component_hint(pci_devices: Path = PCI_DEVICES_DIR) -> str:
    """Suggest the cuda component to NVIDIA users who have not installed it."""
    components = os.environ.get("SNAP_COMPONENTS")
    if components and (Path(components) / CUDA_COMPONENT).is_dir():
        return ""
    if not has_nvidia_gpu(pci_devices):
        return ""
    return f"For NVIDIA GPU acceleration, run: snap install {SNAP_NAME}+{CUDA_COMPONENT}"
