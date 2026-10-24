import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'app'))
import cpu_mdx_cli


class CpuMdxExit(unittest.TestCase):
    def test_normal_success_flushes_before_exit(self):
        events = []
        with patch.object(cpu_mdx_cli, 'print', side_effect=lambda *a, **kw: events.append('stdout'), create=True), \
             patch.object(cpu_mdx_cli.sys.stderr, 'flush', side_effect=lambda: events.append('stderr')), \
             patch.object(cpu_mdx_cli.os, '_exit', side_effect=lambda code: events.append(code)):
            cpu_mdx_cli.finish_cli(lambda: events.append('main returned'))
        self.assertEqual(events, ['main returned', 'stdout', 'stderr', 0])

    def test_success_system_exit(self):
        with patch.object(cpu_mdx_cli.os, '_exit') as exit_, patch('builtins.print'):
            cpu_mdx_cli.finish_cli(Mock(side_effect=SystemExit(0)))
            exit_.assert_called_once_with(0)

    def test_failed_exit_is_never_success(self):
        with patch.object(cpu_mdx_cli.os, '_exit') as exit_:
            with self.assertRaises(SystemExit) as error:
                cpu_mdx_cli.finish_cli(Mock(side_effect=SystemExit(1)))
            self.assertEqual(error.exception.code, 1)
            exit_.assert_not_called()

    def test_exception_is_never_success(self):
        with patch.object(cpu_mdx_cli.os, '_exit') as exit_:
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                cpu_mdx_cli.finish_cli(Mock(side_effect=RuntimeError('inference failed')))
            exit_.assert_not_called()
