from os import path

import setuptools

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

# extract version
path = path.realpath('src/microEye/_version.py')
version_ns = {}
with open(path, encoding='utf8') as f:
    exec(f.read(), {}, version_ns)
version = version_ns['__version__']

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='microeye',
    version=version,
    author='Mohammad Nour Alsamsam',
    author_email='nour.alsamsam@gmail.com',
    description='A python toolkit for fluorescence microscopy \
        that features hardware control, data analysis and vizualization \
        for super-resolution single-molecule localization microscopy and \
        single-partical tracking.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/samhitech/microEye',
    project_urls={
        'miEye Paper': 'https://doi.org/10.1016/j.ohx.2022.e00368',
        'miEye OSF': 'http://doi.org/10.17605/osf.io/j2fqy',
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'microEye = microEye.launcher:main',
        ],
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.9.4, <3.12',
    include_package_data=True,
    install_requires=requirements,
)
