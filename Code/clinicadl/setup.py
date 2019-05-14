from setuptools import setup

setup(
    name = 'clinicadl',
    version = '0.0.1',
    description = 'Deep learning classification with clinica',
    url = '',
    packages = ['clinicadl',],
    entry_points = {
        'console_scripts': [
            'clinicadl = clinicadl.main:main',
        ],
    },
    license = 'MIT',
    author = 'Mauricio Diaz',
    author_email = 'mauricio.diaz@inria.fr',
    )
