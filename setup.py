from __future__ import annotations

from setuptools import find_packages, setup

from sandcat import __version__

setup(
    name='sandcat',
    version=__version__,
    description='Codebase for SandCat molecular dynamics simulations',
    author='SandCat contributors',
    # find_packages() picks up sandcat and all of its subpackages.
    packages=find_packages(exclude=['sandcat.tests', 'sandcat.tests.*']),
    include_package_data=True,
    package_data={
        'sandcat': [
            'forcefields/*.xml',
            'forcefields/LICENSE-*.txt',
            'probes/*.sdf',
        ],
    },
    install_requires=[
        'openmmtools',
        'mpi4py'
    ],
)
