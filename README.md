# Series2Vec
### Update
This is a PyTorch implementation of Similarity-based learning: A contrastive learning inspired self-supervised method for time series
<p align="center">
    <img src="Fig/Series2Vec_01.png">
</p> 

### Get data from UEA Archive and HAR and Ford Challenge
Download dataset files and place them into the specified folder
UEA: http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip
Copy the datasets folder to: Datasets/UEA/

## Setup

_Instructions refer to Unix-based systems (e.g. Linux, MacOS)._

This code has been tested with `Python 3.7` and `3.8`.

`pip install -r requirements.txt`

## Run

To see all command options with explanations, run: `python main.py --help`
In 'configuration.py' you can select the datasets and modify the model parameters.
For example:

`self.parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')`

or you can set the paprameters:

`python main.py --epochs 1500 --data_dir Datasets/UEA/Heartbeat`

