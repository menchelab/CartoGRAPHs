import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '2.0.1'
PACKAGE_NAME = 'cartoGRAPHs'
AUTHOR = 'Chris H.'
AUTHOR_EMAIL = 'chris@menchelab.com'
URL = 'https://github.com/menchelab/cartoGRAPHs'

LICENSE = 'MIT License'
DESCRIPTION = 'A Network Layout and Visualization Package'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'networkx',
        'plotly',
        'colormath',
        'umap-learn',
        'scikit_learn',
        'shapely',
        'colormath',
        'numba',
        'pynndescent==0.5.8'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
