from setuptools import setup

install_requires = [
    'numpy'
]
test_requires = install_requires + []

setup(
    name='mecs',
    license='MIT License',
    version='0.0.1',
    install_requires=install_requires,
    test_requires=test_requires)
