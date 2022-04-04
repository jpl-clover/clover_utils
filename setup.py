from setuptools import setup

setup(
    name='clover_utils',
    version='0.0.1',    
    description='Utility functions and classes for the CLOVER organisation.',
    url='https://github.com/jpl-clover/clover_utils',
    author='CLOVER Team',
    author_email='isaac.r.ward@jpl.nasa.gov',
    packages=['clover_utils'],
    install_requires=[
        'mpi4py>=2.0',
        'numpy',                     
    ]
)