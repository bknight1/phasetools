from setuptools import setup, find_packages

setup(
    name='phasetools',
    version='0.0.1',
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