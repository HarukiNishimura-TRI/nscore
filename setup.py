from setuptools import find_packages, setup

setup(
    name="nscore",
    version="0.0.1",
    description="Sequential statistical hypothesis testing for generalized performance measures.",
    authors=["David Snyder", "Haruki Nishimura"],
    author_emails=["dasnyder@princeton.edu", "haruki.nishimura@tri.global"],
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy>=1.20",
        "scipy",
        "cvxpy",
        "statistical-comparison-core",
        "statistical-comparison-helpers",
    ],
)