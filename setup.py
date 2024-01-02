import setuptools

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='microEye',
    version='1.0.5',
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
        'miEye OSF': 'http://doi.org/10.17605/osf.io/j2fqy'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    python_requires='>=3.9.4',
    include_package_data=True,
    requires = [
        'autopep8 (==2.0.4)',
        'dask (==2022.1.0)',
        'h5py (==3.10.0)',
        'hidapi (==0.14.0)',
        'matplotlib (==3.6.3)',
        'numba (==0.57.1)',
        'numpy (==1.24.4)',
        'ome_types (==0.4.3)',
        'opencv_python (==4.5.3)',
        'pandas (==1.3.3)',
        'pyflakes (==2.4.0)',
        'PyQt5 (==5.15.10)',
        'PyQt5_sip (==12.11.1)',
        'pyqtgraph (==0.13.3)',
        'pyserial (==3.5)',
        'pyueye (==4.96.952)',
        'QDarkStyle (==3.1)',
        'QScintilla (==2.13.4)',
        'scikit_image (==0.18.3)',
        'scikit_learn (==1.1.3)',
        'scipy (==1.11.4)',
        'setuptools (==69.0.2)',
        'tables (==3.9.1)',
        'tifffile (==2022.2.2)',
        'zarr (==2.10.3)',
    ]
)
