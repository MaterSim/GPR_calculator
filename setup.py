from distutils.core import setup
import os

# Set compiler to g++
os.environ['CC'] = 'g++'
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gpr_calc",
    version="0.0.3",
    description="GPR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=[
        "gpr_calc",
        "gpr_calc.kernels",
    ],

    package_data={
        "gpr_calc.kernels": ["*.cpp"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "cffi>=1.0.0",
        "mpi4py>=3.0.3",
        "ase>=3.23.0",
        "pyxtal>=1.0.5",
    ],
    python_requires=">=3.9.1",
    license="MIT",
    cffi_modules=[
        "gpr_calc/kernels/libdot_builder.py:ffibuilder",
        "gpr_calc/kernels/librbf_builder.py:ffibuilder"
    ],
)
