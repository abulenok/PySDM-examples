from setuptools import setup, find_packages

setup(
    name='PySDM-examples',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    install_requires=['PySDM @ git+https://github.com/atmos-cloud-sim-uj/PySDM@f8f52c6#egg=PySDM',
                      'PyMPDATA @ git+https://github.com/atmos-cloud-sim-uj/PyMPDATA@e70e077#egg=PyMPDATA',
                      'pystrict',
                      'matplotlib',
                      'ipywidgets',
                      'ghapi',
                      'evtk @ git+https://github.com/paulo-herrera/PyEVTK@5e0616e'],  # TODO #457
    author='https://github.com/orgs/atmos-cloud-sim-uj/people',
    license="GPL-3.0",
    packages=find_packages(include=['PySDM_examples', 'PySDM_examples.*'])
)
