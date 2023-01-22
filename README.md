# bayesian_active_learning

## Repository for `bayesian_active_learning`, an implementation of ["Deep Bayesian Active Learning with Image Data"](https://arxiv.org/abs/1703.02910) using Haiku and Optax. 

The repository has a module `bayesian_active_learning` which contains:
- `acquisition_functions.py`: acquistion functions and sampling methodology for active learning procedure
- `data_utils.py`: Simple classes for loading datasets from numpy arrays
- `experiment.py`: Contains end to end procedure for active learning setup
- `losses.py`: loss functions (only classification loss implemented for now)
- `metrics.py`: simple methods to evaluate jax models given parameters and a dataloader
- `models.py`: Bayesian convnet implementation and Haiku transformation method
- `training.py`: `fit` method for training haiku models with optax
- `utils.py`: miscellaneous utility functions

A series of notebooks in the `notebooks` folder investigate different aspects of the paper:
- `01_standard_experiment.ipynb`: Partial reconstruction of Figure 1 from the paper, using a subselection of the acquisition functions used in the paper with a relatively large batch size
<p align="center">
  <img height="300" src="https://user-images.githubusercontent.com/19254716/213914382-26500a85-4832-4459-9329-636d3f1ef9e2.png">
</p>

- `02_reduced_batch_size.ipynb`: An experiment using a much lower acquisition size per iteration (20 rather than 100), since using the standard BALD acquisition function can have degraded performance under large acquisition sizes (see ["BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning"
](https://arxiv.org/abs/1906.08158))
<p align="center">
  <img height="300" src="https://user-images.githubusercontent.com/19254716/213914594-7f9905ea-99ac-451d-a917-85835cf452ab.png">
</p>

- `03_modified_pool_set.ipynb`: An experiment where the pool set (unlabelled images that are selected during active learning) is made artifically unbalanced, by removing 75% of the "2"s in the set.
<p align="center">
  <img height="300" src="https://user-images.githubusercontent.com/19254716/213914642-1a641e5f-a9b6-4a48-b2b5-e151c57944e7.png">
</p>

- `04_further_extensions.ipynb`: a short notebook detailing other ideas for future work.
