# Data valuation: The partial ordinal Shapley value for machine learning
Code for implementation of "Data valuation: The partial ordinal Shapley value for machine learning".

This repository builds on the [DataShapley](https://github.com/amiratag/DataShapley) repository 
and contains implementations of TMC, CMC and CTMC algorithm.


## Dataset

- [Wine](https://archive.ics.uci.edu/ml/datasets/Wine)
- [Cancer](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))
- [Adult](https://archive.ics.uci.edu/ml/datasets/Adult)


## Requirments
Python, NumPy, Tensorflow 1.12, Scikit-learn, Matplotlib

## Example command
args:
- dataset_name: Name of dataset (['wine', 'cancer', 'adult_s'])
- results_path: path for saving results 
- tol: Truncated factor
- ratio: ratio for CMC and CTMC
- noise: ratio of noisy label

```
$ python3 main.py --dataset_name cancer --tol 0.05 --start_run 0 --num_run 5 --noise 0.2
```

## Reference
If you find our code useful for your research, please cite our paper.
```
@misc{liu2023data,
      title={Data valuation: The partial ordinal Shapley value for machine learning}, 
      author={Jie Liu and Peizheng Wang and Chao Wu},
      year={2023},
      eprint={2305.01660},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
