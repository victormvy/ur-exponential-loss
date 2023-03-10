# Exponential loss regularisation for encouraging ordinal constraint to shotgun stocks quality assessment

This repository contains the implementation of the exponential regularised categorical cross-entropy loss function for PyTorch.

## Requirements

The following packages are required in order to use this loss function:
- PyTorch
- NumPy
- Scipy

## Execution

The loss function is implemented as a PyTorch loss function. It can be used to optimise any PyTorch model. A usage example is included in the same file. You can run this example from the terminal using the following command:

    python exponential_loss.py

To use the loss function in your code, you can import like:

    from exponential_loss import ExponentialCrossEntropyLoss

## Citation

You can cite this work as follows:

    @article{vargas2023exponential,
        title = {Exponential loss regularisation for encouraging ordinal constraint to shotgun stocks quality assessment},
        journal = {Applied Soft Computing},
        pages = {110191},
        year = {2023},
        issn = {1568-4946},
        doi = {10.1016/j.asoc.2023.110191},
        author = {Víctor Manuel Vargas and Pedro Antonio Gutiérrez and Riccardo Rosati and Luca Romeo and Emanuele Frontoni and César Hervás-Martínez},
    }
