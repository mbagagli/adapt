from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required_list = f.read().splitlines()

setup(
    name="adapt-fw",
    version="0.8.2",
    author="Matteo Bagagli",
    author_email="matteo.bagagli@ingv.it",
    description="ADAptive Picking Toolbox: a framework for seismic picking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ethz.ch/mbagagli",
    python_requires='>=3.6',
    install_requires=required_list,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Topic :: Software Development :: Libraries",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers"
    ],
    setup_requires=['wheel'],
    include_package_data=True,
    zip_safe=False,
    scripts=['bin/adapt_PRODUCTION.py', 'bin/adapt2cnv.py',
             'bin/adapt_JHD-VELEST.py', 'bin/cnv2adapt.py']
)
