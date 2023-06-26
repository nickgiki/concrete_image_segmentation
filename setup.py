from setuptools import find_packages, setup

with open("requirements.txt") as f:
    requirements = [line.rstrip() for line in f.readlines()]

setup(
    name = "concrete_image_segmentation",
    version = "1.0.0",
    author = "Nikos Gkikizas",
    description="Segmentation of 3d concrete sample scans",
    author_email = "nickgikizas@hotmail.com",
    install_requires = requirements,
    packages = find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    # entry_points = (
        
    # ),
)