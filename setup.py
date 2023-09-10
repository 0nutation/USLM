# coding=utf-8
import os
import sys
from pathlib import Path
from subprocess import DEVNULL, PIPE, run

from setuptools import find_packages, setup

project_root = Path(__file__).parent

# modified from https://github.com/lhotse-speech/lhotse/blob/master/setup.py



if sys.version_info < (3,):
    # fmt: off
    print(
        "Python 2 has reached end-of-life and is no longer supported by valle."
    )
    # fmt: on
    sys.exit(-1)

if sys.version_info < (3, 7):
    print(
        "Python 3.6 has reached end-of-life on December 31st, 2021 "
        "and is no longer supported by valle."
    )
    sys.exit(-1)




install_requires = [
    "encodec",
    "phonemizer",
]

try:
    # If the user already installed PyTorch, make sure he has torchaudio too.
    # Otherwise, we'll just install the latest versions from PyPI for the user.
    import torch

    try:
        import torchaudio
    except ImportError:
        raise ValueError(
            "We detected that you have already installed PyTorch, but haven't installed torchaudio. "
            "Unfortunately we can't detect the compatible torchaudio version for you; "
            "you will have to install it manually. "
            "For instructions, please refer either to https://pytorch.org/get-started/locally/ "
            "or https://github.com/pytorch/audio#dependencies"
        )
except ImportError:
    install_requires.extend(["torch", "torchaudio"])

docs_require = (
    (project_root / "requirements.txt").read_text().splitlines()
)
tests_require = [
    # "pytest==7.1.3",
    # "pytest-forked==1.4.0",
    # "pytest-xdist==2.5.0",
    # "pytest-cov==4.0.0",
]
workflow_requires = [""]
dev_requires = sorted(
    docs_require
    + tests_require
    + workflow_requires
    + ["jupyterlab", "matplotlib"]
)
all_requires = sorted(dev_requires)

if os.environ.get("READTHEDOCS", False):
    # When building documentation, omit torchaudio installation and mock it instead.
    # This works around the inability to install libsoundfile1 in read-the-docs env,
    # which caused the documentation builds to silently crash.
    install_requires = [
        req
        for req in install_requires
        if not any(req.startswith(dep) for dep in ["torchaudio", "SoundFile"])
    ]

setup(
    name="USLM",
    version='0.1.0',
    python_requires=">=3.7.0",
    description="USLM: Unified Speech Language Models",
    author="Dong Zhang",
    author_email="dongzhang22@fudan.edu.cn",
    long_description=(project_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="Apache-2.0 License",
    packages=find_packages(exclude=["test", "test.*"]),
    include_package_data=True,
    entry_points={},
    install_requires=install_requires,
    extras_require={
        "docs": docs_require,
        "tests": tests_require,
        "dev": dev_requires,
        "all": all_requires,
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
)
