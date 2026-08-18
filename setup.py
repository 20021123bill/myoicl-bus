from setuptools import find_packages, setup

setup(
    name="myoicl",
    version="0.1.0",
    description=(
        "MyoICL: label-free in-context adaptation for cross-user sEMG decoding"
    ),
    packages=find_packages(),
    package_data={"myoicl": ["configs/*.yaml"]},
    install_requires=[],  # relies on the qwerty conda env (torch, emg2qwerty, ...)
)
