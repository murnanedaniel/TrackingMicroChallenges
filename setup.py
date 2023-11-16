from setuptools import setup, find_packages

setup(
    name='TrackingMicroChallenges',
    version='0.1.0',
    description='Deep Learning MicroChallenges for Particle Tracking',
    author='Daniel Murnane',
    author_email='dtmurnane@lbl.gov',
    url='https://github.com/ExaTrkX/TrackingMicroChallenges',
    package_dir={"microchallenges": "src"},
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'torch',
        'tqdm',
        'class_resolver',
        'torch',
        'torch_geometric',
        'lightning',
        'wandb',
    ],
)