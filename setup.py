from setuptools import setup, find_packages
from darkchem import __version__


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()
    required = None

pkgs = find_packages(exclude=('examples', 'docs', 'resources'))

setup(
    name='darkchem',
    version=__version__,
    description='Deep learning to uncover molecular "dark matter"',
    long_description=readme,
    author='Sean M. Colby',
    author_email='sean.colby@pnnl.gov',
    url='https://github.com/pnnl/darkchem',
    license=license,
    packages=pkgs,
    install_requires=required,
    entry_points={
        'console_scripts': ['darkchem = darkchem.cli:main']
    }
)
