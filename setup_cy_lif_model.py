# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension('cy_lif_model',
        ['cy_lif_model.pyx'],
        include_dirs = [np.get_include()],
        script_args = ['--compiler=mingw32'])
    ]

setup(
    ext_modules = cythonize(extensions)
    )
