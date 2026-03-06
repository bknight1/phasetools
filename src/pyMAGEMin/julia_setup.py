import argparse
import shutil
import subprocess
import sys


def _run_julia(args):
    return subprocess.run(["julia", *args], capture_output=True, text=True)


def check_julia_installed():
    """Return True when the Julia executable is available on PATH."""
    return shutil.which("julia") is not None


def check_magemin_installed():
    """Return True when Julia package MAGEMin_C can be imported."""
    if not check_julia_installed():
        return False
    result = _run_julia(["-e", 'using MAGEMin_C'])
    return result.returncode == 0


def install_magemin():
    """Install MAGEMin_C using Julia's package manager."""
    if not check_julia_installed():
        raise RuntimeError("Julia is not available on PATH.")
    result = _run_julia(["-e", 'using Pkg; Pkg.add("MAGEMin_C")'])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Julia package installation failed.")


def print_status():
    julia_ok = check_julia_installed()
    magemin_ok = check_magemin_installed()

    print(f"Julia installed: {'yes' if julia_ok else 'no'}")
    print(f"MAGEMin_C installed: {'yes' if magemin_ok else 'no'}")

    if not julia_ok:
        print("Install Julia first: https://julialang.org/downloads/")
        return 1

    if not magemin_ok:
        print("Install MAGEMin_C with: pyMAGEMin-julia-setup --install")
        return 2

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check/install Julia + MAGEMin_C for pyMAGEMin."
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
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    if args.check or not args.install:
        return print_status()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
