
## Generating artificial multispectral images using neural style transfer: a study with application in channel alignment
---
### Matheus Vieira da Silva, Leandro H. F. P. Silva, Jocival D. Dias Jr, Mauricio C. Escarpinati, André R. Backes, and João Fernando Mari
---

#### *** Submitted to MDPI Sensors ***
---
---



# Environment configuration

Install Anaconda:
```
https://www.anaconda.com/products/distribution
```

Create and activate an enviroment with Python 3.8:
```
conda create -n env-nst-py38 python=3.8
conda activate env-nst-py38
```

Install Pytorch:
```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
```

Install Matplotlib:
```
pip install matplotlib==3.5.1
```

Install Scikit-image:
```
pip install scikit-image==0.19.2
```

Install Pandas
```
pip install pandas==1.4.1
```

## Observations:

If you don´t hava a compatible GPU installed, install PyTorch as:

```
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly -c pytorch
```

Recomended only for testing purposes. For training a new model is high recomemended to use a GPU with 11 GB RAM, at least.


# Dataset configuration

Our dataset is composed of multispectral images with 5 channels (blue, green, red, red-edge and near-ir). Each channel is stored in a separated gray scale tif image. These images are in the folder ```./data/soybean/```. The sufix indicates the channel (1 - blue, 2 - green, 3 - red, 4 - red-edge and 5 - near-ir). All images have XXX x XXX pixels. The images are resized to XXX to XXX during during the processing to reduce computational cost.

# Hyperparameter searching.

To perform the hipeparameter seaching execute the `search.py` without any comand line argument.
```
(env-nst-py38)$ python search.py
```

It will result in a folder named ```search``` with the results of executing the neural style transfer image generation algoritm varying the parameters ```learning_rate``` and ```number of iterations``` as the table below:

* Learning rate: {0,001, 0.002, ..., 0.009}
* Number of iterations: {1000, 2000, 3000}

In order to select the best results in acordance to the criterium of minimal loss (style loss + content loss) execute the ```match.py``` passing as argument the path to the experiment folder
```
(env-nst-py38)$ python match.py experiments
```

It will search for the combination of hyperparameters which resulted in the minimum combined loss.
In sequence will apply the histogram matching between the generated image and the original style image.

The best artificial images generated by the NST network will be saved in the folder `experiments/optim_results/nst/`, and the final results (after the histogram matching) will be in the folder `experiments/optim_results/matched/`. 
A CSV file will be generated containing information about each image and the best hyperparameter values for each one.

# Training

If you have the CSV file containg the best hyperparameter values, it is not necessary to run the hyperparameter optmization. You can execute only the desired experiment runing the ```train.py``` passing the correct CSV file as argument.
```
(env-nst-py38)$ python train.py experiments/results.csv
```

It will generated a new folder named `experiments_train` with the folders `original`, `nst`, and `matched`. The folder original contains the original misaligned images after resizing, the `nst` folder contains the artificial aligned images generated by the NST algorithm and the `matched` folder will contain the generated images after post-processing.





