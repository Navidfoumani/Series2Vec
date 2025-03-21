# Series2Vec

This is a PyTorch implementation of **Series2Vec: Similarity-based Self-supervised Representation Learning for Time Series Classification**.

### 📅 Code Update: [01.03.2025]
#### ⚠️ Note:
If you downloaded the code prior to the latest update, please ensure to update to the current version as it is consistent with the paper.

<p align="center">
    <img src="Fig/Series2Vec_01.png">
</p> 

---

## 📥 Dataset

The datasets used for training and evaluation can be downloaded from the following locations:

### 1. **Large Benchmark Datasets**  
Download the datasets from [this Google Drive link](https://drive.google.com/drive/folders/1YLdbzwslNkmi3No19C3aGdmfAUSoruzB?usp=sharing).  
After downloading, place them in the `Datasets/Benchmarks/` directory.


### 2. **UEA Archive**  
You can  download it from [the official UEA website](https://www.timeseriesclassification.com/aeon-toolkit/Archives/Multivariate2018_ts.zip).

---

## 📑 References

- **Paper**: [Series2Vec Paper on Springer](https://link.springer.com/article/10.1007/s10618-024-01043-w)
- **Blog Post**: [Meet Series2Vec: A New Way to Decode Time Series](https://www.linkedin.com/pulse/meet-series2vec-navids-new-way-decode-time-t5uzc/?trackingId=FxA0fznaSRKhyRRfSO0t2A%3D%3D)

---

## ⚙️ Setup

_Instructions are for Unix-based systems (e.g., Linux, MacOS)._

To see all command options with explanations, run: `python main.py --help`.
In `utils/args.py` you can select the datasets and modify the model parameters.
For example:

`self.parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')`

or you can set the parameters:

`python main.py --dataset Benchmarks`


## Citation
If you find **Series2vec** useful for your research, please consider citing this paper using the following information:

````
```
@article{series2vec2024,
  title={Series2vec: similarity-based self-supervised representation learning for time series classification},
  author={Foumani, Navid Mohammadi and Tan, Chang Wei and Webb, Geoffrey I and Rezatofighi, Hamid and Salehi, Mahsa},
  journal={Data Mining and Knowledge Discovery},
  volume={38},
  number={4},
  pages={2520--2544},
  year={2024},
  publisher={Springer}
}

```
````
