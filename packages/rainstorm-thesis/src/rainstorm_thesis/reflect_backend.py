"""Export actual backend signatures/schemas inside its isolated worker environment.

Usage: python -m rainstorm_thesis.reflect_backend vame > vame-backend.json
The exported native library schema is evidence; curated plugin schemas govern
editable controls. This command never constructs or fits a model.
"""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path


BACKENDS = {'vame': ('vame', 'vame-py', ['preprocessing', 'create_trainset', 'train_model', 'segment_session']),
    'kpms': ('keypoint_moseq', 'keypoint-moseq', ['format_data', 'fit_pca', 'init_model', 'fit_model', 'apply_model'])}


def export_backend(name):
    module_name, distribution, functions = BACKENDS[name]
    backend = importlib.import_module(module_name)
    apis = {}
    for name in functions:
        function = getattr(backend, name)
        signature = inspect.signature(function)
        source = inspect.getsourcefile(function)
        apis[name] = {'signature': str(signature), 'parameters': [
            {'name': parameter.name, 'kind': parameter.kind.name,
             'annotation': str(parameter.annotation),
             'required': parameter.default is inspect.Parameter.empty}
            for parameter in signature.parameters.values()],
            'source_sha256': 'sha256:' + hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None}
    result = {'version': '1', 'backend': module_name, 'package_version': version(distribution), 'apis': apis}
    if module_name == 'vame':
        from vame.schemas.project import ProjectSchema
        result['backend_schema'] = ProjectSchema.model_json_schema()
    from storm.suite import default_catalog
    from .plugin import register
    catalog = default_catalog(); register(catalog)
    result['components'] = [c for c in catalog.describe() if c['name'].startswith(
        'vame_' if module_name == 'vame' else 'kpms_')]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('backend', choices=BACKENDS)
    args = parser.parse_args()
    print(json.dumps(export_backend(args.backend), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
