import setuptools

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='microEye',
    version='2.3.1',
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
        'Programming Language :: Python :: 3',
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
    install_requires=[
        'autopep8==2.2.0',
        'dask==2024.5.2',
        'h5py==3.11.0',
        'hidapi==0.14.0',
        'matplotlib==3.9.2',
        'numba==0.60.0',
        'numpy==1.26.4',
        'ome-types==0.5.2',
        'opencv-python==4.9.0.80',
        'pandas==2.2.3',
        'pco==2.2.1',
        'pyfiglet==1.0.2',
        'pyflakes==3.2.0',
        'pyjokes==0.6.0',
        'PyOpenGL==3.1.7',
        'pyqtdarktheme==2.1.0',
        'pyqtgraph==0.13.7',
        'pyserial==3.5',
        'PySide6==6.7.2',
        'PySide6-Addons==6.7.2',
        'PySide6-Essentials==6.7.2',
        'pyueye==4.96.952',
        'QDarkStyle==3.2.3',
        'QScintilla==2.14.1',
        'scikit-image==0.22.0',
        'scikit-learn==1.5.0',
        'scipy==1.14.1',
        'setuptools==69.0.2',
        'tables==3.9.1',
        'tabulate==0.9.0',
        'tqdm==4.66.4',
        'tifffile==2024.8.10',
        'vispy==0.14.2',
        'zarr==2.18.2',
    ],
)
