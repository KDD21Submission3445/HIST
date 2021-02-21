# Source code and data for the HIST Framework

### Required packages
The code has been tested running under Python 3.6.5, with the following packages installed (along with their dependencies):
- pytorch == 1.5.0
- numpy == 1.19.1
- pandas == 1.1.4
- tqdm == 4.46.1

### Running the code
Install the [Git Large File Storage (LFS)](https://git-lfs.github.com/) first.
```
git clone https://github.com/KDD21Submission3445/HIST.git
cd HIST
python hist.py
```
The git clone command may take some time because the data file is large.

The csi 500 data is available at [here](https://drive.google.com/file/d/1JlSD3IVwOH0Ts3zjCJnQFK_96_dFkHbb/view?usp=sharing), you can modify the input data (line 672 to 680 in hist.py) to run the code on csi 500.
