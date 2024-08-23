# Copyright (C) 2022 Nusret Ipek

# Import setuptools

try:
    from setuptools import setup, find_packages
    _has_setuptools = True
except ImportError:
    from distutils.core import setup
    
# Setup parameters

DISTNAME = "rhsegmentor"
VERSION = "0.1.0"
AUTHOR = "Jan Verwaeren"
AUTHOR_EMAIL ="Jan.Verwaeren@UGent.be"

DESC = "RHSegmentor: automated root hair segmentation"
with open("README.md", "r") as f:
    LONG_DESC = f.read()
LONG_DESC_TYPE = "text/markdown"

URL = "https://github.com/jverwaer/root_segmentor"
LICENSE = 'MIT'
INSTALL_REQUIRES = [
    "numpy>=1.23.0",
    "pillow>=10.4.0",
    "scikit-image>=0.22.0",
    "scipy>=1.11.4",
    "pandas>=2.1.4",
    "scikit-learn>=1.5.0",
    "joblib>=1.4.2",
    "pylineclip>=1.0.0",
    "matplotlib>=3.8.2"
]

#PACKAGES=["rhsegmentor"]
PACKAGES = find_packages(include=['rhsegmentor'])
CLASSIFIERS = [
              "Development Status :: 4 - Beta",
              "Programming Language :: Python :: 3",
              "Programming Language :: Python :: 3 :: Only",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.8",
              "Programming Language :: Python :: 3.9",
              "License :: OSI Approved :: MIT License",
              "Intended Audience :: Science/Research",
              "Operating System :: Microsoft :: Windows",
              "Operating System :: Unix",
              "Operating System :: MacOS",
]

# Setup

if __name__ == "__main__":
    
    setup(name=DISTNAME,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          description=DESC,
          long_description=LONG_DESC,
          long_description_content_type=LONG_DESC_TYPE,
          license=LICENSE,
          packages=PACKAGES,
          url=URL,
          download_url=URL,
          install_requires=INSTALL_REQUIRES,
          classifiers=CLASSIFIERS,
          include_package_data=True
         )