import os
from setuptools import setup, Extension
import pybind11

# Получаем абсолютный путь к корневой директории проекта
project_root = os.path.abspath(os.path.dirname(__file__))

# Пути к файлам и директориям
source_root = "vector-entropy"
bindings_file = os.path.join(project_root, source_root, "python", "bindings.cpp")
entropy_cpp_file = os.path.join(project_root, source_root, "src", "entropy.cpp")
entropy_include_dir = os.path.join(project_root, source_root, "src")

ext_modules = [
    Extension(
        "entropy_core",
        sources=[bindings_file, entropy_cpp_file],
        include_dirs=[
            pybind11.get_include(),
            entropy_include_dir
        ],
        language='c++',
        extra_compile_args=['-std=c++11'],
    ),
]

setup(
    name="entropy_cpp",
    version="0.1.0",
    author="Andrey Kotelnikov",
    description="Vector entropy calculation utility with Python bindings",
    ext_modules=ext_modules,
    zip_safe=False,
    include_package_data=True,
)
