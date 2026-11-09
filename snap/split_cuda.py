"""Copy the packages a CUDA install adds or replaces over a CPU install.

The result is a site-packages directory for the cuda component; the launcher puts
it ahead of the base snap's site-packages so its torch shadows the CPU build.
"""

import shutil
import sys
from importlib.metadata import distributions
from pathlib import Path

# Distributions present in both installs but built differently per variant.
REPLACED = {"torch", "torchvision"}


def site_packages(python_root: Path) -> Path:
    return next(python_root.glob("lib/python3.*/site-packages"))


def installed(site: Path) -> dict:
    return {dist.metadata["Name"].lower(): dist for dist in distributions(path=[str(site)])}


def top_level_entries(dist) -> set[str]:
    """First path segment of every file the distribution installed into site-packages."""
    entries = {Path(file).parts[0] for file in dist.files or [] if not str(file).startswith("..")}
    return entries - {"__pycache__"}


def main():
    cpu_root, cuda_root, output = (Path(arg) for arg in sys.argv[1:4])
    cpu_site, cuda_site = site_packages(cpu_root), site_packages(cuda_root)
    cpu, cuda = installed(cpu_site), installed(cuda_site)

    for name in REPLACED:
        cpu_version, cuda_version = (d[name].version.split("+")[0] for d in (cpu, cuda))
        if cpu_version != cuda_version:
            sys.exit(f"{name} differs between variants: {cpu_version} (cpu) vs {cuda_version} (cuda)")

    selected = sorted(name for name in cuda if name not in cpu or name in REPLACED)
    entries = set().union(*(top_level_entries(cuda[name]) for name in selected))
    # Shadowing works per top-level package, so the cuda set must not split one with the base.
    kept = set().union(*(top_level_entries(cpu[name]) for name in cpu if name not in REPLACED))
    shared = sorted(entries & kept)
    if shared:
        sys.exit(f"CUDA packages share top-level entries with the base snap: {shared}")

    output.mkdir(parents=True)
    for entry in sorted(entries):
        source = cuda_site / entry
        if source.is_dir():
            shutil.copytree(source, output / entry, symlinks=True)
        elif source.exists():
            shutil.copy2(source, output / entry)
    print(f"cuda component: {', '.join(selected)}")


if __name__ == "__main__":
    main()
