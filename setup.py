""" See https://packaging.python.org/guides/distributing-packages-using-setuptools
"""
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
#
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
#
setup(
    # This is the name of your project. The first time you publish this
    # package, this name will be registered for you. It will determine
    # how users can install this project, e.g.: pip install sampleproject
    # And where it will live on PyPI: https://pypi.org/project/sampleproject/
    #
    name="peer-reviewed-flower",  # Required
    # Versions should comply with PEP 440:
    # https://www.python.org/dev/peps/pep-0440/
    #
    version="0.1.0",  # Required
    # This is a one-line description or tagline of what your project does.
    #
    description="Peer Review Flower: Federated Learning with Peer Review using Flower",  # Optional
    # This is an optional longer description of your project that represents
    # the body of text which users will see when they visit PyPI.
    #
    long_description=long_description,  # Optional
    # Denotes that our long_description is in Markdown.
    #
    long_description_content_type="text/markdown",  # Optional (see note above)
    # This should be a valid link to your project's main homepage.
    #
    url="https://github.com/passerim/peer-reviewed-flower",  # Optional
    # This should be your name or the name of the organization which owns the
    # project.
    #
    author="Mattia Passeri",  # Optional
    # This should be a valid email address corresponding to the author listed
    # above.
    #
    author_email="mattia.passeri2@studio.unibo.it",  # Optional
    # Classifiers help users find your project by categorizing it.
    # For a list of valid classifiers, see https://pypi.org/classifiers/
    #
    classifiers=[  # Optional
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        # Specify the Python versions you support here.
        #
        "Programming Language :: Python :: 3",
    ],
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    packages=find_packages(exclude=["tests", "tests.*"]),  # Required
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match.
    #
    python_requires=">=3.7, <4",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    install_requires=[
        "flwr[simulation]==1.0.0",
        "numpy",
        "overrides",
    ],  # Optional
)
