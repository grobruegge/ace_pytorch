# Automatic Concept-based Explanation and Evaluation

This is the PyTorch implementations of the paper [Towards Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129) presented at NeurIPS 2019.

### Prerequisites

Required python libraries:

```
  - torch
  - torchvision
  - matplotlib
  - sklearn
  - tqdm
  - PIL
```

### Getting started

#### Arguments

- **--target_class**: Target class of the PyTorch network to be explained 
- **--source_dir**: Directory which contains the target class images as well as random images used to train the CAV. The following folders are (minimally) required within `source_dir`:
  - One folder with the name of `target_class` (needs to be changed accordingly) containing images of the target class to be explained. Create a seperate folder for each target_class you wish to explain.
  - One folder called "_random_" (fixed name, do not change!) contaning at least 500 random images, ideally of as many different classes as possible
- **--working_dir**: Directory used to save/load cached values (such as model activations) and results. This folder should be empty if using the framework for the first time.
- **--model_name**: Name of the PyTorch model to be interpreted (has to be recognited by PyTorch).
- **--model_path**: If you want to use a different dataset than ImageNet1k for any model available in PyTorch, you can additionally pass the file path to the model weights for that dataset and they will be loaded. CURRENTLY, THIS DOES NOT WORK!
- **--labels_path** If you cloned this repository, the label-file already exists and this argument should not be changed. Only change if for whatever reason you want to move the label-file to a differrent directory.
- **--num_random_datasets**

- **--num_parallel_runs**: Whether to run the code using multiprocessing and specifies how many CPU cores should be used. Defaults to 0 (i.e., no multiprocessing). CURRENTLY, THIS DOES NOT WORK!

To be continued...