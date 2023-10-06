from setuptools import setup, find_packages

setup(
    name="yubie",
    version="0.0.1",
    packages=find_packages(),
    install_requires = [
        # Required packages
    ],
    entry_points='''
    [console_scripts]
        yubie=main:init
        yubie_benchmark=main:benchmark --name={nameOfBenchmark}
        yubie_test=main:test --name={nameOfTest}
    '''
)
