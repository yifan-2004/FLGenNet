"""Dependency-free checks; these do not import Torch or deserialize checkpoints."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import inspect
import os
from os import path as osp
from pathlib import Path
import random
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('config_paths', ROOT / 'codes/config_paths.py')
PATHS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATHS)


def extract_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class PortabilityTest(unittest.TestCase):
    def test_all_python_files_parse(self):
        for folder in ('codes', 'scripts', 'tests'):
            for path in (ROOT / folder).rglob('*.py'):
                with self.subTest(file=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding='utf-8'))

    def test_path_resolution_preserves_absolute_expands_home_and_none(self):
        self.assertEqual(PATHS.resolve_project_path('datasets/low', ROOT), str(ROOT / 'datasets/low'))
        absolute = str(ROOT.parent / 'external/low')
        self.assertEqual(PATHS.resolve_project_path(absolute, ROOT), absolute)
        self.assertEqual(PATHS.resolve_project_path('~/data', ROOT), osp.expanduser('~/data'))
        self.assertIsNone(PATHS.resolve_project_path(None, ROOT))

    def test_relative_path_is_independent_of_working_directory(self):
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as folder:
                os.chdir(folder)
                self.assertEqual(PATHS.resolve_project_path('experiments/lol2.pth', ROOT),
                                 str(ROOT / 'experiments/lol2.pth'))
        finally:
            os.chdir(previous)

    def test_configuration_parse_resolves_all_phases_and_checkpoints(self):
        source = ROOT / 'codes/utils/options.py'
        fixture = {
            'name': 'fixture',
            'datasets': {'test_1': {'dataroot_gt': '/external/gt', 'dataroot_lq': 'data/low'},
                         'val': {'dataroot_gt': None, 'dataroot_lq': 'data/val'}},
            'path': {'pretrain_network_g': 'experiments/lol2.pth', 'resume_state': None}}
        namespace = {'__file__': str(source), 'osp': osp,
                     'resolve_project_path': PATHS.resolve_project_path,
                     'yaml': SimpleNamespace(load=lambda stream, Loader: deepcopy(fixture)),
                     'ordered_yaml': lambda: (object(), object())}
        parse = extract_function(source, 'parse', namespace)
        for training in (True, False):
            result = parse(ROOT / 'options/test/model/test.yml', is_train=training)
            self.assertEqual(result['path']['root'], str(ROOT))
            self.assertEqual(result['path']['pretrain_network_g'], str(ROOT / 'experiments/lol2.pth'))
            self.assertEqual(result['datasets']['test_1']['dataroot_gt'], '/external/gt')
            self.assertEqual(result['datasets']['test_1']['dataroot_lq'], str(ROOT / 'data/low'))
            self.assertEqual(result['datasets']['test_1']['phase'], 'test')
            self.assertIsNone(result['datasets']['val']['dataroot_gt'])
            self.assertIsNone(result['path']['resume_state'])
            key = 'experiments_root' if training else 'results_root'
            self.assertTrue(result['path'][key].startswith(str(ROOT)))

    def test_cli_default_paths_and_absolute_override(self):
        source = ROOT / 'codes/train.py'
        calls = []
        def parse(path, is_train):
            calls.append((path, is_train))
            return {'manual_seed': 10}
        namespace = {'__file__': str(source), 'argparse': argparse, 'osp': osp,
                     'resolve_project_path': PATHS.resolve_project_path, 'parse': parse,
                     'get_dist_info': lambda: (0, 1), 'set_random_seed': lambda seed: None,
                     'random': random}
        parse_options = extract_function(source, 'parse_options', namespace)
        for training in (True, False):
            with patch.object(sys, 'argv', ['entrypoint']):
                result = parse_options(is_train=training)
            expected = 'options/train/model/FLGNet.yml' if training else 'options/test/model/test.yml'
            self.assertEqual(calls[-1], (str(ROOT / expected), training))
            self.assertFalse(result['dist'])
        with patch.object(sys, 'argv', ['entrypoint', '-opt', '/external/custom.yml']):
            parse_options(is_train=False)
        self.assertEqual(calls[-1][0], '/external/custom.yml')

    def test_architecture_scan_discovers_flg_and_instantiates(self):
        source = ROOT / 'codes/models/archs/__init__.py'
        tree = ast.parse(source.read_text())
        body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        class Dummy:
            def __init__(self, **kwargs):
                self.options = kwargs
        imported = []
        def fake_import(name):
            imported.append(name)
            return SimpleNamespace(FLGenNet=Dummy) if name.endswith('.FLG') else SimpleNamespace()
        namespace = {'__file__': str(source), 'osp': osp,
                     'scandir': lambda folder: [p.name for p in Path(folder).iterdir() if p.is_file()],
                     'importlib': SimpleNamespace(import_module=fake_import)}
        exec(compile(ast.Module(body=body, type_ignores=[]), str(source), 'exec'), namespace)
        self.assertIn('codes.models.archs.FLG', imported)
        network = namespace['define_network']({'type': 'FLGenNet', 'nrb_low': 5})
        self.assertEqual(network.options, {'nrb_low': 5})
        with self.assertRaisesRegex(ValueError, 'not found'):
            namespace['dynamic_instantiation']([], 'Missing', {})

    def test_yaml_selected_classes_exist_in_source(self):
        def class_names(path):
            return {node.name for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
        configurations = [
            ('options/train/model/FLGNet.yml', 'codes/models/lptn_model.py'),
            ('options/test/model/test.yml', 'codes/models/lptn_test_model.py')]
        architecture_classes = class_names(ROOT / 'codes/models/archs/FLG.py')
        for config_name, model_path in configurations:
            config = (ROOT / config_name).read_text()
            model = re.search(r'^model_type:\s*(\S+)', config, re.M).group(1)
            network = re.search(r'^network_g:\s*\n\s+type:\s*(\S+)', config, re.M).group(1)
            self.assertIn(model, class_names(ROOT / model_path))
            self.assertIn(network, architecture_classes)
        test_config = (ROOT / 'options/test/model/test.yml').read_text()
        checkpoint = re.search(r'^\s+pretrain_network_g:\s*(\S+)', test_config, re.M).group(1)
        self.assertTrue((ROOT / checkpoint).is_file())

    def test_checkpoint_loader_prefers_weights_only_when_supported(self):
        source = ROOT / 'codes/models/base_model.py'
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseModel')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'load_network')
        method.decorator_list = []
        for supports_weights_only in (True, False):
            calls = []
            loaded = []
            def modern_load(path, *, weights_only=False):
                calls.append((path, weights_only))
                return {'stub': True}
            def legacy_load(path):
                calls.append((path, None))
                return {'stub': True}
            namespace = {'inspect': inspect,
                         'torch': SimpleNamespace(load=modern_load if supports_weights_only else legacy_load),
                         'logger': SimpleNamespace(info=lambda message: None)}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
            network = SimpleNamespace(load_state_dict=lambda state, strict: loaded.append((state, strict)))
            model = SimpleNamespace(get_bare_model=lambda net: net)
            namespace['load_network'](model, network, 'fake-checkpoint.pth', strict=True)
            self.assertEqual(calls, [('fake-checkpoint.pth', True if supports_weights_only else None)])
            self.assertEqual(loaded, [({'stub': True}, True)])

    def test_local_import_targets_exist(self):
        for path in (ROOT / 'codes').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('codes.'):
                    target = ROOT.joinpath(*node.module.split('.'))
                    self.assertTrue(target.is_dir() or target.with_suffix('.py').is_file(),
                                    f'{path}: missing module {node.module}')


if __name__ == '__main__':
    unittest.main()
