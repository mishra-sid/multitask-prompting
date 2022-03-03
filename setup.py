"""
Scripts to train and evaluate models for soft prompting in multi task scenarios
"""
from setuptools import find_packages, setup

setup(
    name="multitask-prompting",
    version="0.0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*","tests.*", "tests"]),
    package_dir={"multitask-prompting": ""},
    description="Scripts to train and evaluate models for soft prompting in multi task scenarios",
    install_requires=[],
    extras_require={
        "test": ["pytest"],
    },
    entry_points={"console_scripts": ["multitask-prompting=multitask_prompting.train:main"]},
)