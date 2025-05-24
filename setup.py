from setuptools import setup, find_packages

setup(
    name="binance-bot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "plotly>=5.3.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "websockets>=10.0",
        "aiohttp>=3.8.0",
        "python-binance>=1.0.15",
        "pymongo>=4.6.0",
        "motor>=3.3.0",
        "pytest>=6.2.5",
        "pytest-asyncio>=0.16.0",
        "pytest-cov>=4.1.0"
    ],
    python_requires=">=3.9",
) 