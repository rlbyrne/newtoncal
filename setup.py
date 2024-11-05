from setuptools import setup

setup(
    name="Calico",
    author="Ruby Byrne",
    author_email="rbyrne@caltech.edu",
    url="https://github.com/rlbyrne/calico",
    scripts=[
        "calico/calibration_wrappers.py",
        "calico/calibration_optimization.py",
        "calico/cost_function_calculations.py",
        "calico/calibration_qa.py",
        "calico/caldata.py",
    ],
)
