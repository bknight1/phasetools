from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import sys

def check_julia():
    """
    Verify if Julia is installed on the system by checking its version.
    """
    try:
        subprocess.check_call(['julia', '--version'])
        return True
    except Exception:
        return False

def check_magemin():
    """
    Verify if the MAGEMin_C Julia package is installed by checking its status.
    """
    try:
        subprocess.check_call(['julia', '-e', 'using Pkg; Pkg.status("MAGEMin_C")'])
        return True
    except subprocess.CalledProcessError:
        return False

def install_magemin():
    """
    Install the MAGEMin_C Julia package using Julia's package manager.
    """
    try:
        subprocess.check_call(['julia', '-e', 'using Pkg; Pkg.add("MAGEMin_C")'])
    except Exception as e:
        raise RuntimeError(f"Installation failed: {e}")

class CustomInstall(install):
    def run(self):
        if not check_julia():
            self.announce("Julia not found. Please install Julia before proceeding.", level=4)
            sys.exit(1)
        else:
            self.announce("Julia is installed.", level=2)

        if not check_magemin():
            self.announce("MAGEMin_C package not found. Installing...", level=2)
            try:
                install_magemin()
                self.announce("MAGEMin_C package installed successfully.", level=2)
            except Exception as e:
                self.announce(f"Error installing MAGEMin_C package: {e}", level=4)
                sys.exit(1)
        else:
            self.announce("MAGEMin_C package is already installed.", level=2)

        # Proceed with the standard installation process
        install.run(self)

setup(
    name='pyMAGEMin',
    version='0.0.1',
    author='Ben Knight',
    author_email='ben.knight@curtin.edu.au',
    description='A python package to perform MAGEMin calculations in python',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    cmdclass={'install': CustomInstall},
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
        'scipy>=1.4.0',
        'molmass>=2024.0.0',
        'juliacall>=0.9',
    ],
    license='AFL-3.0',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        # Remove the deprecated 'License :: OSI Approved :: Academic Free License (AFL)' classifier.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Natural Language :: English',
    ],
    python_requires='>=3.10',
)