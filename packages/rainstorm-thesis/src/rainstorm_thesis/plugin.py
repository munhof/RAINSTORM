"""Register RAINSTORM pose adapters with the STORM execution worker."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from storm.suite import Component
from storm.contracts import ModelInputContract
from .preparation import resolve_preparation_steps

from .data import load_dlc_csv, load_dlc_h5
from .example_benchmarks import nor_test_benchmark
from .metrics import register_metrics
from .preprocessing import (
    LikelihoodFilter,
    OrientPose,
    RecenterPose,
    SelectPoseCoordinates,
    TemporalPoseWindows,
)
from .vame_models import VAMEOfficialModel, VAMENativeModel
from .visualizations import RainstormPoseTimeline, RainstormStateTimeline


RAINSTORM_ROOT = Path(__file__).resolve().parents[4]
logger = logging.getLogger(__name__)


def _build_supervised_model(config):
    """Load inference dependencies only inside the scientific worker."""
    from rainstorm.models.supervised import RainstormSupervisedModel

    return RainstormSupervisedModel(config)


def register(catalog) -> None:
    """Add H5 ingestion, pose preparation, VAME models, and SVG views."""
    catalog.preparation_resolver = resolve_preparation_steps
    catalog.connectors["dlc_h5"] = load_dlc_h5
    catalog.connectors["dlc_csv"] = load_dlc_csv
    for step in (SelectPoseCoordinates, RecenterPose, OrientPose,
                 LikelihoodFilter, TemporalPoseWindows):
        catalog.steps.register(step)
    for visualization in (RainstormPoseTimeline, RainstormStateTimeline):
        catalog.visualizations.register(visualization)
    try:
        dataset_preset = nor_test_benchmark()
    except (OSError, ValueError) as error:
        logger.warning('Skipping optional NOR example benchmark: %s', error)
    else:
        catalog.register_dataset_preset(dataset_preset)
    register_metrics(catalog.metrics)
    catalog.register(Component(
        "vame_native",
        VAMENativeModel,
        ("group", "infer", "checkpoint"),
        {
            "type": "object",
            "properties": {
                "n_states": {"type": "integer", "minimum": 2, "default": 50},
                "latent_dim": {"type": "integer", "minimum": 1, "default": 20},
                "hidden_dim": {"type": "integer", "minimum": 1, "default": 128},
                "epochs": {"type": "integer", "minimum": 1, "default": 25},
                "batch_size": {"type": "integer", "minimum": 1, "default": 256},
                "learning_rate": {"type": "number", "minimum": 0.0000001,
                                  "default": 0.001},
                "kld_weight": {"type": "number", "minimum": 0, "default": 0.5},
                "seed": {"type": "integer", "minimum": 0, "default": 156},
                "device": {
                    "type": "string", "default": "auto",
                    "enum": ["auto", "cpu", "cuda"],
                    "description": (
                        "Auto usa GPU si PyTorch la detecta; cuda también selecciona "
                        "GPU AMD con ROCm."
                    ),
                },
                "window_storage": {"type": "string", "default": "memory",
                                   "enum": ["memory", "mapped"],
                                   "description": "mapped guarda las ventanas temporalmente en disco y las lee por lotes."},
                "rnn_backend": {"type": "string", "default": "auto",
                                "enum": ["auto", "native"],
                                "description": "native evita MIOpen/cuDNN en las GRU; conserva cómputo GPU."},
                "num_threads": {"type": "integer", "minimum": 1, "default": 2},
            },
        },
        input_contract=ModelInputContract(input_type="temporal_pose", preparation="external",
                                          required_steps=("pose.temporal_windows",)),
    ))
    catalog.register(Component(
        "vame_official",
        VAMEOfficialModel,
        ("group", "infer"),
        {
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "default": "rainstorm_vame_official"},
                "working_directory": {"type": "string",
                                      "default": ".storm/rainstorm/vame_official"},
                "pose_paths": {
                    "type": "array",
                    "default": [],
                    "description": (
                        "Se completan desde las sesiones de pose registradas; "
                        "dejá la lista vacía para usar las fuentes del estudio."
                    ),
                },
                "videos": {
                    "type": "array",
                    "default": [],
                    "description": (
                        "Se vinculan a los videos de esas sesiones; dejá la lista "
                        "vacía si el estudio no tiene videos."
                    ),
                },
                "source_software": {"type": "string", "default": "DeepLabCut",
                                    "enum": ["DeepLabCut", "SLEAP", "LightningPose", "NWB"]},
                "fps": {"type": "number", "minimum": 0.001, "default": 30.0},
                "segmentation_algorithm": {"type": "string", "default": "hmm",
                                           "enum": ["hmm", "kmeans"]},
                "centered_reference_keypoint": {
                    "type": "string", "default": "body",
                    "description": "Punto corporal que usa el preprocesado interno de VAME.",
                },
                "orientation_reference_keypoint": {
                    "type": "string", "default": "nose",
                    "description": "Referencia de orientación del preprocesado interno de VAME.",
                },
                "config_kwargs": {
                    "type": "object",
                    "description": (
                        "Parámetros de entrenamiento, preprocesado y segmentación de VAME. "
                        "Este adapter usa su preprocesado interno y no acepta pasos "
                        "codeless adicionales de STORM."
                    ),
                    "default": {
                        "n_clusters": 50,
                        "model_snapshot": 10,
                        "model_convergence": 10,
                        "time_window": 19,
                        "zdims": 10,
                        "max_epochs": 50,
                        "batch_size": 32,
                        "confidence": 0.60,
                        "seed": 156,
                    },
                },
            },
        },
        input_contract=ModelInputContract(input_type="pose_sources", preparation="internal",
                                          granularity="session"),
    ))
    catalog.register_recipe_preset({
        "id": "vame_native_pose_ego",
        "name": "VAME nativo · pose_ego",
        "model": "vame_native",
        "config": {
            "n_states": 50,
            "latent_dim": 20,
            "epochs": 25,
            "batch_size": 256,
            "kld_weight": 0.5,
            "seed": 156,
        },
        "steps": [
            {
                "type": "pose.select_coordinates",
                "config": {"names": [
                    name for part in (
                        "nose", "left_ear", "right_ear", "head", "neck", "body",
                        "left_shoulder", "right_shoulder", "left_midside",
                        "right_midside", "left_hip", "right_hip", "tail_base",
                        "tail_mid", "tail_end",
                    ) for name in (f"{part}_x", f"{part}_y")
                ]},
            },
            {
                "type": "pose.likelihood_filter",
                "config": {"threshold": 0.6, "bodyparts": [
                    "nose", "left_ear", "right_ear", "head", "neck", "body",
                    "left_shoulder", "right_shoulder", "left_midside",
                    "right_midside", "left_hip", "right_hip", "tail_base",
                    "tail_mid", "tail_end",
                ]},
            },
            {
                "type": "pose.recenter",
                "config": {"center_bodypart": "body", "bodyparts": [
                    "nose", "left_ear", "right_ear", "head", "neck", "body",
                    "left_shoulder", "right_shoulder", "left_midside",
                    "right_midside", "left_hip", "right_hip", "tail_base",
                    "tail_mid", "tail_end",
                ]},
            },
            {
                "type": "pose.orient_coordinates",
                "config": {
                    "from_bodypart": "nose",
                    "toward_bodypart": "body",
                    "target_angle_degrees": 90,
                },
            },
            {
                "type": "pose.temporal_windows",
                "config": {"offsets": list(range(-9, 11))},
            },
        ],
        "description": (
            "Reconstrucción de pose_ego: selecciona 15 puntos, aplica el umbral "
            "histórico de confianza (0,6), centra en body, alinea nose-body a 90° "
            "y crea ventanas de 9 frames anteriores a 10 posteriores. La receta "
            "actual no incluye el filtrado anatómico/por velocidad ni el suavizado "
            "mediano de 5 frames de Tesis_Facu; esos pasos aún no están disponibles "
            "en este worker."
        ),
        "provenance": (
            "Parámetros del apéndice D.4/D.5, transcritos en "
            "docs/reconstruction/recipe-reference.json. "
            "Los pasos de preparación se reconstruyeron desde "
            "Tesis_Facu/src/rainstorm/experiments/study/input_builder.py. "
            "Tesis_Facu/notebooks/model_experiments/08_final_vame_official_train_test.py "
            "identifica la variante pose_ego; no se recuperaron los pesos originales."
        ),
        "status": (
            "Reconstrucción parcial con etapas compatibles disponibles; faltan "
            "filtrado anatómico/por velocidad y suavizado mediano. Los pesos, "
            "resultados y el reentrenamiento histórico no están verificados."
        ),
    })
    catalog.register_recipe_preset({
        "id": "vame_official_ego_roi",
        "name": "VAME oficial · ego_roi",
        "model": "vame_official",
        "config": {
            "project_name": "rainstorm_vame_official_ego_roi",
            "segmentation_algorithm": "kmeans",
            "centered_reference_keypoint": "body",
            "orientation_reference_keypoint": "nose",
            "config_kwargs": {
                "n_clusters": 50,
                "model_snapshot": 10,
                "model_convergence": 10,
                "time_window": 19,
                "zdims": 10,
                "max_epochs": 50,
                "batch_size": 32,
                "confidence": 0.60,
                "learning_rate": 0.0005,
                "seed": 156,
            },
        },
        "steps": [],
        "description": (
            "VAME aplica su preprocesado egocéntrico interno. La referencia ROI "
            "describe análisis contextual posterior; no agrega una etapa de "
            "entrenamiento en esta receta."
        ),
        "provenance": (
            "Parámetros del apéndice D.4/D.5, transcritos en "
            "docs/reconstruction/recipe-reference.json. "
            "Tesis_Facu/notebooks/model_experiments/08_final_vame_official_train_test.py "
            "documenta el flujo final; no se recuperaron los pesos originales."
        ),
        "status": (
            "Receta reconstruida; los resultados y el reentrenamiento histórico "
            "no están verificados."
        ),
    })
    supervised_runtime = os.environ.get(
        "RAINSTORM_SUPERVISED_PYTHON",
        "/opt/rainstorm/supervised-runtime/bin/python",
    )
    for name, filename in (
        ("supervised_simple", "example_simple.keras"),
        ("supervised_wide", "example_wide.keras"),
    ):
        catalog.register(Component(
            name,
            _build_supervised_model,
            ("infer",),
            {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "default": str(
                            RAINSTORM_ROOT
                            / "examples"
                            / "models"
                            / "trained_models"
                            / filename
                        ),
                    },
                    "runtime_python": {"type": "string", "default": supervised_runtime},
                    "threshold": {"type": "number", "minimum": 0, "maximum": 1,
                                  "default": 0.5},
                    "output_meaning": {
                        "type": "string",
                        "default": "unknown",
                        "description": (
                            "Categoría positiva. Para comparar con etiquetas humanas, "
                            "debe coincidir exactamente con la segunda categoría de la taxonomía."
                        ),
                    },
                    "negative_output_meaning": {
                        "type": "string",
                        "default": "unknown",
                        "description": (
                            "Categoría negativa; debe coincidir con la primera categoría "
                            "de una taxonomía binaria para habilitar el ranking."
                        ),
                    },
                    "category_mapping_version": {
                        "type": "string",
                        "default": "1",
                        "description": "Versión reproducible de la correspondencia de clases 0 y 1.",
                    },
                    "training_population": {"type": "string", "default": "unknown"},
                },
            },
            version="1",
            input_contract=ModelInputContract(input_type="features", preparation="external",
                shape=(12,) if name == "supervised_simple" else (7, 12)),
        ))
