import argparse
import importlib
import os
import shutil
import subprocess
import sys

# MAGEMin_C's dependency stack relies on WeakRefStrings' legacy PosLen API;
# WeakRefStrings 1.4.3 requires Parsers 2.x and fails with Parsers 3.x.
_PARSERS_VERSION = "2.8.3"
_WEAKREFSTRINGS_VERSION = "1.4.3"


def _juliacall_context():
    """Return the Julia executable and project selected by juliacall/juliapkg."""
    try:
        juliapkg = importlib.import_module("juliapkg")
        project = juliapkg.project()
        executable = juliapkg.executable()
    except Exception as exc:
        raise RuntimeError(
            "Could not determine juliacall's Julia project and executable. "
            "Ensure juliacall/juliapkg are installed and configured."
        ) from exc

    if not project:
        raise RuntimeError("juliapkg did not provide a Julia project for juliacall.")
    if not executable:
        raise RuntimeError("juliapkg did not provide a Julia executable for juliacall.")

    project = os.path.abspath(os.fspath(project))
    if not os.path.isdir(project):
        raise RuntimeError(f"juliacall's Julia project does not exist: {project}")
    return os.fspath(executable), project


def _run_julia(args, project=None, executable=None):
    if project is None or executable is None:
        resolved_executable, resolved_project = _juliacall_context()
        executable = executable or resolved_executable
        project = project or resolved_project
    return subprocess.run(
        [executable, f"--project={project}", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def check_julia_installed():
    """Return True when the Julia executable is available on PATH."""
    return shutil.which("julia") is not None


def check_magemin_installed():
    """Return True when MAGEMin_C loads through the phasetools Python bridge."""
    if not check_julia_installed():
        return False

    try:
        _executable, project = _juliacall_context()
    except RuntimeError as exc:
        print(f"Could not check MAGEMin_C: {exc}", file=sys.stderr)
        return False

    bridge_check = (
        "from phasetools import MAGEMin_C; "
        "assert MAGEMin_C is not None; "
        "assert callable(MAGEMin_C.Initialize_MAGEMin)"
    )
    env = os.environ.copy()
    env["JULIA_PROJECT"] = project
    try:
        result = subprocess.run(
            [sys.executable, "-c", bridge_check],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except OSError:
        return False
    return result.returncode == 0


def install_magemin():
    """Install MAGEMin_C with compatible Parsers and WeakRefStrings releases."""
    if not check_julia_installed():
        raise RuntimeError("Julia is not available on PATH.")

    executable, project = _juliacall_context()

    setup = (
        "using Pkg; "
        'Pkg.add([PackageSpec(name="MAGEMin_C"), '
        f'PackageSpec(name="Parsers", version="{_PARSERS_VERSION}"), '
        f'PackageSpec(name="WeakRefStrings", version="{_WEAKREFSTRINGS_VERSION}")]); '
        f'Pkg.pin(PackageSpec(name="Parsers", version="{_PARSERS_VERSION}")); '
        f'Pkg.pin(PackageSpec(name="WeakRefStrings", version="{_WEAKREFSTRINGS_VERSION}"))'
    )
    result = _run_julia(["-e", setup], project=project, executable=executable)
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or "Julia package installation failed.")

    # Pkg can return successfully while reporting a dependency precompile
    # failure. Verify the actual package import in a fresh Julia process.
    result = _run_julia(
        ["-e", "using MAGEMin_C"], project=project, executable=executable
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(details or "MAGEMin_C could not be loaded after installation.")


def print_status():
    julia_ok = check_julia_installed()
    magemin_ok = check_magemin_installed()

    print(f"Julia installed: {'yes' if julia_ok else 'no'}")
    print(f"MAGEMin_C installed: {'yes' if magemin_ok else 'no'}")

    if not julia_ok:
        print("Install Julia first: https://julialang.org/downloads/")
        return 1

    if not magemin_ok:
        print("Install MAGEMin_C with: phasetools-julia-setup --install")
        return 2

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check/install Julia + MAGEMin_C for phasetools."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether Julia and MAGEMin_C are available.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install MAGEMin_C through Julia Pkg.",
    )
    args = parser.parse_args(argv)

    if args.install:
        try:
            install_magemin()
            print("MAGEMin_C installed successfully.")
        except (RuntimeError, OSError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.check or not args.install:
        return print_status()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
