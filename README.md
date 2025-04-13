# boltz
Mode jumping for deep Boltzmann machines. 
Applies the teleworking walkers algorithm seen in https://arxiv.org/abs/2106.02686.

    .
    ├── demos   # Demonstration notebooks for a selection of models
    ├── tests   # Test root
    └── boltz   # Code root


## Instructions for Ubuntu/OSX

0. Install the generic dependencies: `Python 3.11`, `uv`, `git`:

1. Define your project root (`[project]`) and navigate there:

    ```shell
    mkdir [project]
    cd [project]
    ```

2. Clone the repository:

    ```shell
    git clone https://github.com/timsf/boltz.git
    ```

3. Start the `Jupyter` server:

    ```shell
    uv run jupyter notebook
    ```

4. Access the `Jupyter` server in your browser and navigate to the notebook of interest.