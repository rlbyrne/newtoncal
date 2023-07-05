from setuptools import setup

setup(
    name="NewCal",
    author="Ruby Byrne",
    author_email="rbyrne@caltech.edu",
    url="https://github.com/rlbyrne/newcal",
    scripts=["newcal/calibration.py", "newcal/cost_function_calculations.py"],
)
