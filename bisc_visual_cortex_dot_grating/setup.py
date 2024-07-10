from setuptools import setup, find_packages


with open('bisc/VERSION.txt', 'r') as f:
    VERSION = f.readline().split('"')[1]

setup(
    name="bisc",
    version=VERSION,
    author='Zhe Li',
    python_requires='>=3.9',
    packages=find_packages(),
    package_data={'bisc': ['VERSION.txt']},
    install_requires=['h5py', 'jarvis>=0.7'],
)
