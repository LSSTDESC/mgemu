from setuptools import setup, find_packages


PACKAGENAME = "mgemu"
VERSION = "0.0.dev"


setup(
    name="mgemu",
    version=1.0,
    author="Georgios Valogiannis, Nesar Ramachandra, Mustapha Ishak, Katrin Heitmann",
    author_email="gvalogiannis@g.harvard.edu",
    description="Emulator for fast generation of boost in matter power spectra for f(R) Hu-Sawicki model",
    long_description="We present a Gaussian Process Emulator for interpolation in cosmological parameters and across redshifts for the approximating of boost in the power spectra due to Modified Gravity (MG) effects.",
    install_requires=[
                    "numpy",
                    "scikit-learn",
                    "tensorflow==1.14.0",
                    "gpflow==1.5.1",
    ],
    packages=find_packages(),
    package_data={"mgemu": ["models/*"]},
    url="https://github.com/LSSTDESC/{}".format("mgemu"),
)
