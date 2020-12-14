# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyaudiodsptools',
    version='1.0.0',
    description='Package for Audio Processing with Numpy',
    long_description=readme,
    author='Arjaan Auinger',
    author_email='arjaan.auinger@gmail.com',
    url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)