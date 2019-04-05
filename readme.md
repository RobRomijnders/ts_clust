# Clustering and Visualizing time series using auto encoders
By Ivan Sekulic and Rob Romijnders

# Getting started
Get the code from remote
```bash
git clone https://github.com/RobRomijnders/ts_clust.git
cd ts_clust
```

Set up the python virtual environment
```bash
pipenv install
pipenv shell
```

Run the following scripts:

  * `main.py` for training the auto encoder and saving the model in `log_tb/`. During training, the script prints three losses. If all goes well, they should all three at least go below 1.0.
  * `extract_representations.py` for extracting the representations for a given data set. This loads a saved model from the `log_tb/` directory. This script saves the representations in `output_rep/`
  * `plot_representations.py` a small script to plot the representations in `output_rep/`. This script also shows how to use the representations for other uses.
  
  
## Download the data
Download the data from the [UCR time series data base](https://www.cs.ucr.edu/~eamonn/time_series_data/)

Put the data in the `ts_clust/data/UCR_datasets` directory.

# Introduction

In this project, we set out to visualize and cluster time series. We have many many time series, but they have no label and we don't know how they relate to eachother. An auto encoder can compress the complex time series to a small latent code. A latent code captures all the variablility in each time series. 

Another problem is to cluster the data. In our big pile of time series, some time series naturally cluster together. These time series are similar in some semantic sense. Therefore, we run a clustering algorithm on our latent codes. As the latent codes capture the variability in the time series. A cluster of latent codes will represent a semantically similar set of time series.

In this project, we make the following contributions
  
  * We present an auto encoder structure to compress time series. We use a compression loss as defined in the Wasserstein Auto encoder
  * We visualize some of the UCR time series and show how the latent codes of the auto encoder capture semantics of the time series.
  * We employ a clustering algorithm on the latent code and show the accuracy of these clusters against ground truth labels. (Note that our entire model is unsupervised)
  
# Data sets

TODO: mention somewhere that we use a few small data sets. Maybe mention the sample size for each data set that we use? 

