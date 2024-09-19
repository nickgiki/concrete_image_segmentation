# `concrete_image_segmentation`
A package that uses the [UNET](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical) neural network architecture to segment 3D grayscale images of concrete samples.

## Project Setup

To set up the project locally follow these steps.

### Prerequisites

Make sure you have the following tools installed on your system:

- [Git](https://git-scm.com/) (Version control system)
- [miniconda](https://docs.anaconda.com/miniconda/) (Software dependency management system)

### Installation

1. **Clone the Repository:**
    Clone the repo and navigate to the folder

    ```bash
    git clone https://github.com/nickgiki/concrete_image_segmentation.git

    cd concrete_image_segmentation
    ```
2. **Create and activate a new conda environment:**
    Create a new conda environment with python 3.11

    ```anaconda
    conda create -n concrete_image_segmentation python=3.10
    ```

    Follow the instructions to complete the installation.

    After the installation is completed activate the new environment

    ```bash
    conda activate concrete_image_segmentation
    ```

3. **Install the package and dependencies:**

    Install the `concrete_image_segmentation` package and its dependencies.

    ```bash
    pip install .
    ```

