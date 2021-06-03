# Continual Learning with Latent Generative Replay 

This repository includes the source code used for evaluating the Latent Generative Replay (LGR) continual learning strategy described in the MPhil dissertation titled *"Improving Continual Learning Resource Efficiency with Latent Generative Replay"*. The repository has been created by forking and extending the [continual learning repository by van de Ven](https://github.com/GMvandeVen/continual-learning). 

## Requirements 

The `pip` environment used for running the experiments has been exported as `requirements.txt`. 

## Files

On top of the additional continual learning repository, we have added/modified the following files/folders: 
* `analysis` - this includes Jupyter Notebooks used for pre-processing and visualising results, creating plots and tables for the report. 
* `experiments` - this includes the scripts for running the experiments as well as the logs from the experiments. 
* `autoencoder_latent.py` - a VAE generative model generating latent replay layer activations. (The implementation is identical to the default implementation provided in `vae_models.py` but it takes and generates 1-dimensional vectors instead of 2-dimensional images). 
* `CKPlusPreprocessing.ipynb` and `FaceCropping.ipynb` - notebooks used for pre-processing CK+ images and cropping them. 
* `CL_metrics_CLAIR.py` - contains code for extracting system metrics. This code has been provided by [Vincenzo Lomonaco et al](https://arxiv.org/abs/2104.00405). 
* `cnn_classifier.py` - the LeNet classifier adapted from [the sample implementation in the PyTorch repository](https://github.com/pytorch/examples/blob/master/mnist/main.py). `cnn_top_classifier.py` and `cnn_root_classifier.py` respectively define the top and the root part of the architecture. 
* `data.py` - we have modified this file to add support for the CK+ and AffectNet datasets. 
* `main.py` - the main scripts for running the experiments. We have modified this to support LGR and the new datasets. We have also added several new flags described below. 
* `naive_rehearsal.py` - this includes our implementation of the replay buffer used to support Naïve Rehearsal and Latent Replay continual learning strategies. 
* `preprocess_affectnet.py` - pre-processes the AffectNet dataset to create the down-sampled or a fully-balanced subset and store it according to the file structure defined by torchvision's [`DatasetFolder` class](https://pytorch.org/vision/stable/datasets.html#datasetfolder). 
* `train_latent.py` - a modification of the default `train.py` allowing the main model to be trained using latent replay. 
* `vgg_classifier.py` - implementation of the VGG-16 architecture. 

## Flags

We have also added a few flags that can be passed as options when running the `main.py` script. Those include: 
* `--latent-replay=(on|off)` - turns latent replay on or off. This flag is set to `on` for Latent Replay, Latent Generative Replay and Latent Generative Replay with Distillation. 
* `--network=(mlp|cnn)` - what type of classifier should be used? 
* `--latent_size` - the size of the latent replay layer (denoted as G<sub>OUT</sub> in the dissertation). 
* `--pretrain-baseline` - pretrain the root of the network on an unseen split of the training data (used for the MNIST experiments only). 
* `--data-augmentation` - enable data augmentations (for CK+ and AffectNet only). Augmentations include random horizontal flip and random rotation. 
* `--vgg-root` - use a pre-trained VGG-16 root/feature extractor. 
* `--buffer-size` - the size of the replay buffer for the rehearsal strategies. 
* `--early-stop` - enable early stopping. 
* `--validation` - validate performance after each task (i.e. obtain and log performance on the entire training dataset and the entire validation dataset). 
* `--use-vgg-face` - use VGG-Face over VGG-16. 

The other flags are documented in [the README file of van de Ven's continual learning repository](https://github.com/GMvandeVen/continual-learning). For example, to run Latent Generative Replay with CK+ on Task-IL (using the VGG-16 architecture), you can run the following command from the root directory of the repository: 
```
./main.py --time --scenario=task --experiment=splitCKPLUS --tasks=4 --vgg-root --network=cnn --iters=2000 --batch=32 --lr=0.0001 --latent-size=4096 --replay=generative --latent-replay=on --g-fc-uni=200 --distill
```
