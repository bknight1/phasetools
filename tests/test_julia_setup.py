import sys
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from phasetools import julia_setup
from phasetools.julia_setup import _PARSERS_VERSION, _WEAKREFSTRINGS_VERSION


class TestJuliaSetup(unittest.TestCase):
    @patch("phasetools.julia_setup.subprocess.run")
    def test_run_julia_targets_explicit_project(self, run_process):
        run_process.return_value = SimpleNamespace(returncode=0, stderr="", stdout="")

        julia_setup._run_julia(
            ["-e", "using MAGEMin_C"],
            project="/julia/envs/juliacall",
            executable="/julia/bin/julia",
        )

        args, kwargs = run_process.call_args
        self.assertEqual(
            args[0],
            [
                "/julia/bin/julia",
                "--project=/julia/envs/juliacall",
                "-e",
                "using MAGEMin_C",
            ],
        )
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertFalse(kwargs["check"])

    @patch(
        "phasetools.julia_setup._juliacall_context",
        return_value=("/julia/bin/julia", "/julia/envs/juliacall"),
    )
    @patch("phasetools.julia_setup._run_julia")
    @patch("phasetools.julia_setup.check_julia_installed", return_value=True)
    def test_install_pins_compatible_julia_dependencies_and_verifies_import(
        self, _check_julia, run_julia, _juliacall_context
    ):
        run_julia.side_effect = [
            SimpleNamespace(returncode=0, stderr="", stdout=""),
            SimpleNamespace(returncode=0, stderr="", stdout=""),
        ]

        julia_setup.install_magemin()

        self.assertEqual(run_julia.call_count, 2)
        install_command = run_julia.call_args_list[0].args[0]
        self.assertEqual(
            run_julia.call_args_list[0],
            call(
                ["-e", unittest.mock.ANY],
                project="/julia/envs/juliacall",
                executable="/julia/bin/julia",
            ),
        )
        self.assertIn('Pkg.add([PackageSpec(name="MAGEMin_C")', install_command[1])
        self.assertIn(
            f'name="Parsers", version="{_PARSERS_VERSION}"', install_command[1]
        )
        self.assertIn(
            f'name="WeakRefStrings", version="{_WEAKREFSTRINGS_VERSION}"',
            install_command[1],
        )
        self.assertIn(
            f'Pkg.pin(PackageSpec(name="Parsers", version="{_PARSERS_VERSION}"))',
            install_command[1],
        )
        self.assertIn(
            "Pkg.pin(PackageSpec(name=\"WeakRefStrings\", "
            f'version="{_WEAKREFSTRINGS_VERSION}"))',
            install_command[1],
        )
        self.assertEqual(
            run_julia.call_args_list[1],
            call(
                ["-e", "using MAGEMin_C"],
                project="/julia/envs/juliacall",
                executable="/julia/bin/julia",
            ),
        )

    @patch(
        "phasetools.julia_setup._juliacall_context",
        return_value=("/julia/bin/julia", "/julia/envs/juliacall"),
    )
    @patch("phasetools.julia_setup.subprocess.run")
    @patch("phasetools.julia_setup.check_julia_installed", return_value=True)
    def test_check_magemin_uses_python_bridge(
        self, _check_julia, run_python, _juliacall_context
    ):
        run_python.return_value = SimpleNamespace(returncode=0, stderr="", stdout="")

        self.assertTrue(julia_setup.check_magemin_installed())

        args, kwargs = run_python.call_args
        self.assertEqual(args[0][:2], [sys.executable, "-c"])
        self.assertIn("from phasetools import MAGEMin_C", args[0][2])
        self.assertIn("MAGEMin_C.Initialize_MAGEMin", args[0][2])
        self.assertEqual(kwargs["env"]["JULIA_PROJECT"], "/julia/envs/juliacall")
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertFalse(kwargs["check"])

    @patch(
        "phasetools.julia_setup._juliacall_context",
        side_effect=RuntimeError("no juliacall project"),
    )
    @patch("phasetools.julia_setup.check_julia_installed", return_value=True)
    def test_check_fails_when_juliacall_project_cannot_be_resolved(
        self, _check_julia, _juliacall_context
    ):
        with patch("phasetools.julia_setup.subprocess.run") as run_python:
            self.assertFalse(julia_setup.check_magemin_installed())

        run_python.assert_not_called()

    @patch(
        "phasetools.julia_setup._juliacall_context",
        return_value=("/julia/bin/julia", "/julia/envs/juliacall"),
    )
    @patch("phasetools.julia_setup._run_julia")
    @patch("phasetools.julia_setup.check_julia_installed", return_value=True)
    def test_install_fails_if_magemin_import_fails_after_pkg_add(
        self, _check_julia, run_julia, _juliacall_context
    ):
        run_julia.side_effect = [
            SimpleNamespace(returncode=0, stderr="", stdout=""),
            SimpleNamespace(
                returncode=1,
                stderr="UndefVarError: PosLen not defined in WeakRefStrings",
                stdout="",
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, "PosLen not defined"):
            julia_setup.install_magemin()


if __name__ == "__main__":
    unittest.main()
