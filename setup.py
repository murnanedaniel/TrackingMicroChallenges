from setuptools import setup, find_packages

setup(
    name='ExaTrkX-DeepGeometry',
    version='0.1',
    description='Deep Learning for Particle Tracking',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/ExaTrkX-DeepGeometry',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'torch',
        'tqdm',
        # Add other dependencies as needed
    ],
)
