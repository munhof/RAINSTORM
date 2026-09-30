from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_vame_worker_lock_is_scoped_to_linux_cpu_deployment():
    project = tomllib.loads(
        (ROOT / "envs" / "vame_worker" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert project["tool"]["uv"]["environments"] == ["sys_platform == 'linux'"]
    assert project["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-cpu"
    assert project["tool"]["uv"]["required-version"] == "==0.11.8"
    containerfile = (ROOT / "Containerfile.storm-worker").read_text(encoding="utf-8")
    assert "uv==0.11.8" in containerfile
    assert "UV_PYTHON_INSTALL_DIR=/opt/uv/python" in containerfile


def test_cuda_worker_has_a_separate_cuda_lock_and_podman_gpu_overlay():
    cuda_env = ROOT / "envs" / "vame_worker_cuda"
    assert (cuda_env / "pyproject.toml").is_file()
    assert (cuda_env / "uv.lock").is_file()

    cuda_project = tomllib.loads(
        (cuda_env / "pyproject.toml").read_text(encoding="utf-8")
    )
    cuda_lock = tomllib.loads((cuda_env / "uv.lock").read_text(encoding="utf-8"))
    cuda_torch = [package for package in cuda_lock["package"] if package["name"] == "torch"]
    cpu_project = tomllib.loads(
        (ROOT / "envs" / "vame_worker" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert cuda_project["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-cu130"
    assert cuda_torch[0]["version"].endswith("+cu130")
    assert cuda_torch[0]["source"]["registry"] == "https://download.pytorch.org/whl/cu130"
    assert cpu_project["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-cpu"

    containerfile = (ROOT / "Containerfile.storm-worker.cuda").read_text(
        encoding="utf-8"
    )
    compose = (ROOT / "compose.storm-plugin.cuda.yaml").read_text(encoding="utf-8")
    assert "envs/vame_worker_cuda/uv.lock" in containerfile
    assert "uv sync --locked" in containerfile
    assert "driver: nvidia" in compose
    assert "count: all" in compose
    assert "capabilities: [gpu]" in compose


def test_rocm_worker_has_a_separate_rocm_lock_and_podman_device_overlay():
    rocm_env = ROOT / "envs" / "vame_worker_rocm"
    assert (rocm_env / "pyproject.toml").is_file()
    assert (rocm_env / "uv.lock").is_file()

    rocm_project = tomllib.loads(
        (rocm_env / "pyproject.toml").read_text(encoding="utf-8")
    )
    rocm_lock = tomllib.loads((rocm_env / "uv.lock").read_text(encoding="utf-8"))
    torch = [package for package in rocm_lock["package"] if package["name"] == "torch"]

    assert rocm_project["tool"]["uv"]["sources"]["torch"]["index"] == "pytorch-rocm"
    assert rocm_project["tool"]["uv"]["sources"]["triton"]["index"] == "pytorch-rocm"
    assert "torch[device-gfx1103]==2.13.0+rocm10.0.0" in rocm_project["project"]["dependencies"]
    assert torch[0]["version"].startswith("2.13.0+rocm10.0.0")
    assert torch[0]["source"]["registry"].rstrip("/") == (
        "https://stable.repo.amd.com/rocm/whl-next"
    )

    containerfile = (ROOT / "Containerfile.storm-worker.rocm").read_text(
        encoding="utf-8"
    )
    compose = (ROOT / "compose.storm-plugin.rocm.yaml").read_text(encoding="utf-8")
    assert "rocm10.0_ubuntu24.04_py3.12_pytorch_release_2.13.0" in containerfile
    assert "uv sync --locked --inexact" in containerfile
    assert "/dev/kfd:/dev/kfd" in compose
    assert "/dev/dri:/dev/dri" in compose
    assert 'group_add: ["keep-groups"]' in compose


def test_pyproject_is_uv_ready_with_isolated_extras():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    storm_project = tomllib.loads(
        (ROOT / "packages" / "rainstorm-thesis" / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]
    storm_dependencies = storm_project["project"]["dependencies"]
    storm_extras = storm_project["project"]["optional-dependencies"]
    uv_sources = storm_project["tool"]["uv"]["sources"]

    assert "storm-traceable" not in dependencies
    assert not any("tensorflow" in dependency for dependency in dependencies)
    assert any("tensorflow" in dependency for dependency in extras["supervised"])
    assert "storm-traceable" in storm_dependencies
    assert storm_project["project"]["requires-python"].startswith(">=3.11")
    assert "vame" in storm_extras
    assert "kpms" in storm_extras
    assert "notebooks" in storm_extras
    assert uv_sources["storm-traceable"]["path"].startswith(
        "../../../STORM-System"
    )


def test_repository_uses_src_notebooks_packages_layout():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert (ROOT / "src" / "rainstorm" / "backend").is_dir()
    assert (ROOT / "src" / "rainstorm" / "frontend").is_dir()
    assert (ROOT / "src" / "rainstorm" / "models").is_dir()
    assert (ROOT / "notebooks" / "original" / "2a-Prepare_positions.ipynb").is_file()
    assert (ROOT / "packages" / "rainstorm-thesis").is_dir()
    assert not (ROOT / "backend").exists()
    assert not (ROOT / "2a-Prepare_positions.ipynb").exists()
    assert project["tool"]["setuptools"]["package-dir"] == {"": "src"}


def test_rainstorm_public_package_keeps_backend_compatibility():
    package = (ROOT / "src" / "rainstorm" / "__init__.py").read_text(
        encoding="utf-8"
    )

    assert "backend" in package
    assert "__path__" in package
    assert "tensorflow" not in package


def test_thesis_domain_module_exposes_storm_loaders_and_specs():
    thesis_package = ROOT / "packages" / "rainstorm-thesis" / "src" / "rainstorm_thesis"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in thesis_package.glob("*.py")
    )

    assert "PoseDatasetConfig" in source
    assert "load_pose_table" in source
    assert "build_pose_dataset_loader" in source
    assert "build_segmentation_study_spec" in source
    assert "build_supervised_study_spec" in source
    assert "storm" in source


def test_storm_notebooks_cover_vame_kpms_and_supervised_workflows():
    notebook_dir = ROOT / "notebooks" / "storm_workflows"
    expected = {
        "01_vame_storm_workflow.ipynb": ("vame", "build_segmentation_study_spec"),
        "02_kpms_storm_workflow.ipynb": ("kpms", "build_segmentation_study_spec"),
        "03_supervised_storm_workflow.ipynb": (
            "supervised",
            "build_supervised_study_spec",
        ),
    }

    for notebook_name, needles in expected.items():
        notebook = json.loads((notebook_dir / notebook_name).read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "load_pose_table" in source
        assert "build_pose_dataset_loader" in source
        assert "StudySpec" in source
        for needle in needles:
            assert needle in source
