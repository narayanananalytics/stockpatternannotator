from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="stockpatternannotator",
    version="0.1.0",
    author="Stock Pattern Annotator Contributors",
    description="OHLC candlestick pattern annotation tool using vectorbtpro",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/narayanananalytics/stockpatternannotator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "vectorbtpro>=1.0.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
    ],
)
