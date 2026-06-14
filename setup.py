from setuptools import setup, find_packages

setup(
    name="machine-failure-classifier",
    version="1.0.0",
    description="Klasifikasi Jenis Kerusakan Mesin — CI/CD/CT Pipeline",
    author="DataMining Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "imbalanced-learn>=0.11.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "flake8>=6.0.0",
            "black>=23.0.0",
        ],
    },
)
