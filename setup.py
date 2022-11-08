import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microEye",
    version="0.2.9",
    author="Mohammad Nour Alsamsam",
    author_email="nour.alsamsam@gmail.com",
    description="A python toolkit for fluorescence microscopy \
        that features multi IDS uEye industrial-grade CMOS \
        cameras and Integrated Optics laser combiner MatchBox.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samhitech/microEye",
    project_urls={
        'miEye Paper': "https://doi.org/10.1016/j.ohx.2022.e00368",
        'miEye OSF': "http://doi.org/10.17605/osf.io/j2fqy"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9.4",
    include_package_data=True
)
