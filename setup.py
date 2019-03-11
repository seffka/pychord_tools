from distutils.core import setup

setup(
    name='pychord_tools',
    version='0.1',
    packages=['pychord_tools'],
    url='',
    license='',
    author='Vsevolod Eremenko',
    author_email='',
    description='',
    package_data={
        'pychord_tools': ['*.pkl'],
    },
    install_requires = [
        'argparse',
        'numpy',
        'vamp',
        'joblib',
        'sklearn',
        'matplotlib',
        'scipy',
        'pyacoustid',
        'dirichlet',
        'musicbrainzngs',
        'essentia'
    ],
    dependency_links=[
        "git+ssh://git@github.com:ericsuh/dirichlet.git"
    ]
)
