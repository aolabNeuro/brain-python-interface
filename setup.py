import setuptools

setuptools.setup(
    name="aolab-bmi3d",
    version="1.2.4",
    author="Lots of people",
    description="electrophysiology experimental rig library",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "django==4.1",
        "celery",
        "jinja2",
        "scipy",
        "numpy==1.26",
        "traits",
        "pandas",
        "patsy",
        "statsmodels",
        "pygame==2.5.0",
        "PyOpenGL",
        "pylibftdi",
        "sphinx",
        "numpydoc",
        "tornado",
        "tables",
        "h5py",
        "pymysql",
        "matplotlib",
        "pyfirmata",
        "hdfwriter==0.1.4",
    ]
)
