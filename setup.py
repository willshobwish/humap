# Author: Wilson Estécio Marcílio Júnior <wilson_jr@outlook.com>
#
# License: BSD 3 clause

from setuptools import setup, Extension, find_packages

from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import os
import sys 
from pathlib import Path

__version__ = "0.2.9"

with open('README.md', 'r') as f:
	long_description = f.read()

ext_modules = None 

def eigen_include_dirs():
    candidates = []

    for env_var in ("EIGEN3_INCLUDE_DIR", "EIGEN_INCLUDE_DIR"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    prefix = Path(sys.prefix)
    candidates.extend([
        prefix / "Library" / "include" / "eigen3",
        prefix / "Library" / "include",
        prefix / "include" / "eigen3",
        prefix / "include",
        Path("C:/Eigen"),
        Path("C:/Eigen/eigen3"),
        Path("/usr/local/include/eigen3"),
        Path("/usr/local/include"),
        Path("/usr/include/eigen3"),
    ])

    include_dirs = []
    seen = set()
    for candidate in candidates:
        if (candidate / "Eigen" / "Core").exists():
            resolved = str(candidate)
            if resolved not in seen:
                include_dirs.append(resolved)
                seen.add(resolved)

    return include_dirs

eigen_dirs = eigen_include_dirs()

if sys.platform == 'win32':
    print("Compiling for Windows")
    ext_modules = [
        Pybind11Extension("_hierarchical_umap",
            ["src/cpp/external/efanna/index.cpp", "src/cpp/external/efanna/index_graph.cpp", "src/cpp/external/efanna/index_kdtree.cpp", "src/cpp/external/efanna/index_random.cpp", "src/cpp/utils.cpp", "src/cpp/umap.cpp", "src/cpp/hierarchical_umap.cpp", "src/cpp/humap_binding.cpp"],
            language='c++',
            include_dirs=eigen_dirs,
            extra_compile_args = [ '/openmp',  '/DINFO'],
            extra_link_args = [],
            define_macros = [('VERSION_INFO', __version__)],
            cxx_std=11,
            ),

    ]
elif sys.platform == 'darwin':
    print("Compiling for MacOS")
    ext_modules = [
    Pybind11Extension("_hierarchical_umap",
        ["src/cpp/external/efanna/index.cpp", "src/cpp/external/efanna/index_graph.cpp", "src/cpp/external/efanna/index_kdtree.cpp", "src/cpp/external/efanna/index_random.cpp", "src/cpp/utils.cpp", "src/cpp/umap.cpp", "src/cpp/hierarchical_umap.cpp", "src/cpp/humap_binding.cpp"],
        language='c++',
        include_dirs=eigen_dirs,
        extra_compile_args = ['-O3', '-std=c++11', '-fPIC', '-fopenmp',  '-DINFO'],
        extra_link_args = ['-O3', '-std=c++11', '-fPIC', '-fopenmp', '-DINFO'],
        define_macros = [('VERSION_INFO', __version__)],
        ),

    ]
else:
    print("Compiling for Linux")
    ext_modules = [
    Pybind11Extension("_hierarchical_umap",
        ["src/cpp/external/efanna/index.cpp", "src/cpp/external/efanna/index_graph.cpp", "src/cpp/external/efanna/index_kdtree.cpp", "src/cpp/external/efanna/index_random.cpp", "src/cpp/utils.cpp", "src/cpp/umap.cpp", "src/cpp/hierarchical_umap.cpp", "src/cpp/humap_binding.cpp"],
        language='c++',
        include_dirs=eigen_dirs,
        extra_compile_args = ['-O3', '-shared', '-std=c++11', '-fPIC', '-fopenmp', '-DINFO'],
        extra_link_args = ['-O3', '-shared', '-std=c++11', '-fPIC', '-fopenmp', '-DINFO'],
        define_macros = [('VERSION_INFO', __version__)],
        ),

    ]


setup(
    name="humap",
    version=__version__,
    author="Wilson E. Marcílio-Jr",
    author_email="wilson_jr@outlook.com",
    url="https://github.com/wilsonjr/humap",
    description="Hierarchical Uniform Manifold Approximation and Projection",
    long_description=long_description,
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    license='MIT',
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.3",
        "scikit-learn>=1.1.3",
        "pynndescent>=0.5.8",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "test": ["pytest"],
        "plot": ["matplotlib"],
    },
    cmdclass={"build_ext": build_ext},
    packages=['humap'],
    zip_safe=False,
)
