from setuptools import setup, find_packages

setup(
    name="yubie",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "ultralytics"
    ],
    entry_points="""
    [console_scripts]
        yubie-dev=yubie:dev
        yubie-test=yubie:test
    """
)
