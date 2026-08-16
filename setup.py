from setuptools import setup, find_packages

setup(
    name='phasetools',
    version='0.1.0',
    author='Ben Knight',
    author_email='ben.knight@curtin.edu.au',
    description='A python package to perform MAGEMin calculations in python',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'matplotlib>=3.0.0',
        'scipy>=1.4.0',
        'molmass>=2024.0.0',
        'juliacall>=0.9',
    ],
    entry_points={
        'console_scripts': [
            'phasetools-julia-setup=phasetools.julia_setup:main',
        ],
    },
    license='LGPL-3.0-or-later',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
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