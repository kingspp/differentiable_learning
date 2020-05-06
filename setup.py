import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = fh.read().split("\n")

setuptools.setup(
    name="difflr",  # Replace with your own username
    version="0.0.1",
    author="kingspp",
    author_email="kingspprathyush@gmail.com",
    description="Differentiable Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kingspp/differentiable_learning",
    install_requires=requirements,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
