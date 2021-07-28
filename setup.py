import setuptools

setuptools.setup(
    name="anomaly-detection",
    version="0.0.1",
    author="gregorius.prasetyo",
    author_email="gregvidy@gmail.com",
    description="Anomaly Detection",
    long_description_content_type="text/markdown",
    url="https://github.com/gregvidy/anomaly-detection",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points="""
        [console_scripts]
        anomaly-detection=src.cli:app
    """,
    install_requires=[
        "numpy==1.21.0",
        "pandas==1.1.4",
        "scikit-learn==0.23.2",
        "shellingham==1.3.2",
        "typer==0.3.2",
        "tqdm==4.56.0",
        "black==21.7b0",
    ],
    python_requires=">=3.7",
)
