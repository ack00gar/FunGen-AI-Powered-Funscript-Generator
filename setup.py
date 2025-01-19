from setuptools import setup, find_packages

setup(
    name="VR funscript creator",
    version="0.0.1",  # Replace with your project version
    author="k00gar",
    description="Create funscripts for VR files",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ack00gar/VR-Funscript-AI-Generator",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # TODO
    ],
    entry_points={
        "console_scripts": [
            # TODO
        ],
    },
    python_requires="=3.11",
)
