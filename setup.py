from setuptools import setup, find_packages

setup(
    name="cebra_ethan",
    version="0.1.0",
    description="CEBRA-Ethan: 个人改进版的CEBRA库，用于神经科学数据的自监督学习和嵌入",
    author="Ethan",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/CEBRA-Ethan",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        # "torch>=1.7.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
)
