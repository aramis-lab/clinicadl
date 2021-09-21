from os.path import abspath, dirname, join, pardir

from setuptools import find_packages, setup

with open(join(dirname(__file__), "clinicadl/VERSION"), "rb") as f:
    version = f.read().decode("ascii").strip()

this_directory = abspath(dirname(__file__))
with open(join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="clinicadl",
    version=version,
    description="Deep learning classification with clinica",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aramis-lab/clinicadl",
    license="MIT license",
    author="ARAMIS Lab",
    maintainer="Mauricio DIAZ",
    maintainer_email="mauricio.diaz@inria.fr",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    data_files=["clinicadl/resources/config/train_config.toml"],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "clinicadl = clinicadl.cmdline:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
    ],
    install_requires=[
        "numpy>=1.17",
        "clinica==0.4.1",
        "torch>=1.8",
        "tensorboard",
        "toml",
        "click>=7.0",
        "pynvml",
    ],
    python_requires=">=3.7",
)
