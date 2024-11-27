# Research Scaffold

Useful tools for running experiments defined by yaml configs in a flexible and extensible manner.

Import `execute_experiments` to get started.

# Example explained

The main package part of the research scaffold is designed to be flexible.
It is intended to be used as demonstrated in the example directory.

To run this, first navigate to inside the example directory and then...

* Use `python main.py -c configs/my_config.yaml` to run a configuration file
    * This encodes the function to call, wandb project to log to if applicable, and any (kw)args for the function to be called with
* Use `python main.py -m meta_configs/my_meta_config.yaml` to run a meta-configuration file
    * These allow for running of several configurations in sequence, doing repeat runs (with automatic rng seed incrementation), or running combinations of configurations
    * Meta configs can also be used to compose configs together, this can allow for specification of a base config file, and then other separate config files that test specific setups in a way that isolates the information being altered
* There are some other command line options, see `python main.py -h` for details
