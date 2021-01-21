

# WaQuaPred

Project for modeling water quality in by using hydrometerological and microbial data. The code repository contains code for processing raw data and making analysis using multiple different models. The main part of the code for analysis is included in the notebooks with some utility methods in python files.

### Data

Data files exists in three different types. The cleaned version is used as input to analysis.

 - **data/raw** : Raw data files recieved from SMHI and GKV.
 - **data/allFiles** : Parsed and formatted data into structured CSV files.
 - **data/cleanedFiles** : Cleaned data with numerical values.
 - **data/exploration** : Plots from data exploration.
 - **data/output** : Results and plots from analysis.

### Files

 - **precprocessing/** : Files for processing raw data and preparing cleaned input.
 - **deprecated/** : Old files not used for final analysis.
 
 - **utils.py** : Helper functions.
 - **data.py** : Methods for reading and parsing input files. One method exist for each input feature.

 - **notebooks/data_exploration.pynb** : Used for exploring and plotting different features.
 - **notebooks/regression_baselines.pynb** : Analysis notebook for producing baselines (Naive, ExpSmooth, ARIMA).
 - **notebooks/regression_var.pynb** : Analysis notebook for producing VAR model.
 - **notebooks/regression_ml.pynb** : Analysis notebook for producing ML models (LASSO, Random Forest, TPOT).

### Run

1. Prepare data with scripts in **preprocessing**:

	- **1_generateInput_GKV.py**
	- **1_generateInput_SMHI.py**
	- **2_cleanInput.py**

2. Run notebooks to perform analysis.

### Other

Developed using Ubuntu 18.04 and Python 3.7. See external libraries in **environment.yml**.

