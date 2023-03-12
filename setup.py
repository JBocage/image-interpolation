from setuptools import setup, find_packages

setup(
    name='image-interp',
    version='1.0.0',
    author='J&M',
    author_email='jnm@gmail.com',
    description='Description of my package',
    packages=find_packages(exclude=["sandbox", "data", "venv"]),    
    install_requires=[],
)