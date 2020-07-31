from os.path import dirname, join, abspath, pardir
from setuptools import setup, find_packages

with open(join(dirname(__file__), 'clinicadl/VERSION'), 'rb') as f:
    version = f.read().decode('ascii').strip()

this_directory = abspath(dirname(__file__))
with open(join(this_directory, pardir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
        name = 'clinicadl',
        version = version,
        description = 'Deep learning classification with clinica',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url = 'https://github.com/aramis-lab/AD-DL',
        license = 'MIT license',
        author = 'ARAMIS Lab',
        maintainer = 'Mauricio DIAZ',
        maintainer_email = 'mauricio.diaz@inria.fr',
        packages = find_packages(exclude=('tests', 'tests.*')),
        include_package_data=True,
        zip_safe=False,
        entry_points = {
            'console_scripts': [
                'clinicadl = clinicadl.main:main',
                ],
            },
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: End Users/Desktop',
            'Intended Audience :: Developers',
            'Programming Language :: Python',
            ],
        install_requires=["numpy>=1.17", "clinica>=0.3.4", "tensorboardX"],
        python_requires='>=3.6',
        )
