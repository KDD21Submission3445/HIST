# Source code and data for the HIST Framework

### Required packages
The code has been tested running under Python 3.6.10, with the following packages installed (along with their dependencies):
- pytorch == 1.5.0
- numpy == 1.19.1
- pandas == 1.1.4
- tqdm == 4.46.1

### Files in the folder
- `data/`
  - `csi300_07to19_30days.pkl`: the stock features (stock price and volume data) and labels;
  - `csi300_stock2concept_matrix.npy`: the adjacent matrix between stocks and predefined concepts;
  - `csi300_date_index.npy`: the index of each date's adjacent matrix between stocks and predefined concepts;

### Data Description
In the csi300_07to19_30days.pkl, for each stock on each date, there are 180-dimensional stock features, which are the opening price, closing price, highest price, lowest price, volume weighted average price (VWAP), and trading volume in the past 30 days. The column 'NoSuspension' indicates whether a stock in suspension. The column 'LABEL0' is the original stock trend label, and the column 'LABEL1' is the stock label after applying normalization on 'LABEL0' of the same date. The column 'MarketValue' is the market capitalization of each stock on each date.

![](https://github.com/KDD21Submission3445/HIST/blob/master/data_example.jpg)

### Running the code
Install the [Git Large File Storage (LFS)](https://git-lfs.github.com/) first.
```
git clone https://github.com/KDD21Submission3445/HIST.git
cd HIST
python hist.py
```
The git clone command may take some time because the data file is large.

The csi 500 data is available at [here](https://drive.google.com/file/d/1JlSD3IVwOH0Ts3zjCJnQFK_96_dFkHbb/view?usp=sharing), you can modify the input data (line 671 to 679 in hist.py) to run the code on csi 500.
