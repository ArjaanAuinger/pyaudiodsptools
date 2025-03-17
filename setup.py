# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('Twine_Readme.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyAudioDspTools',
    version='0.8.3',
    description='Package for audio processing with Numpy',
    long_description=readme,
    author='Arjaan Auinger',
    author_email='arjaan.auinger@gmail.com',
    url='https://github.com/ArjaanAuinger/pyaudiodsptools',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=['numpy'],
    python_requires='>=3.6'
)