"""The setup script for installing the package."""
from setuptools import setup, find_packages


# read the README as a string
with open('README.md') as readme:
    README = readme.read()


# start the setup procedure
setup(
    name='keras_pyramid_pooling_module',
    version='1.1.8',
    description='The Pyramid Pooling Module for Keras.',
    long_description=README,
    long_description_content_type='text/markdown',
    keywords=' '.join(['Keras', 'Pyramid-Pooling-Module', 'Layer']),
    classifiers=[
        'License :: Free For Educational Use',
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    url='https://github.com/Kautenja/keras-pyramid-pooling-module',
    author='Christian Kauten',
    author_email='kautencreations@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*']),
    install_requires=['Keras>=2.4.3', 'tensorflow>=2.4.1'],
)
