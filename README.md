# ivadomed-profile-analysis
Set of analysis tools to help profile IvadoMed, a deep learning medical imaging pipeline, and determine its usability, scalability, and efficiency.

This repository includes all of the scripts we wrote for this project, as well as the results of the various experiments we conducted. The configuration files for each experiment 
is also included here.

# Installation
The steps here will help a user install our forked version of ivadomed to get our scripts in the same directory and be able to execute.

1. Clone our forked version of ivadomed.
```
git clone https://github.com/ivadomed-profile-analysis-project/ivadomed.git
```
2. Setup conda environment and install ivadomed module.
```
cd ivadomed
conda env create --file environment.yml
conda activate IvadoMedEnv
pip install -e .
```
3. Install all dependencies required by ivadomed and for our project.
```
pip install -r requirements_common.txt
pip install -r requirements_dev.txt
pip install -r requirements.txt
pip install -r requirements_gpu.txt
conda install psutil
```
4. Download initial data set used in experiments. 
```
ivadomed_download_data -d data_example_spinegeneric
```
5. Copy-paste the data set folder and rename it. Change the names in dummy_scale.py and run it to generate a data set with N samples; N must be indicated within the script (for 
now).
```
python dummy_scale.py
```

# Usage
It is important to note that paths must be changed both for each configuration file for a particular experiment, as well as within each script. For the configuration json files,
there are only two path variables to be changed:
- "path_output"
- "path_data"

Within each of our scripts, path variables are located at the beginning of main(), so changing paths should be relatively convenient.

Running each experiment is done as shown:
```
conda activate IvadoMedEnv
cd ivadomed
cd ivadomed
cd config

python multexp_batch.py
python multexp_class.py
python multexp_class_scale.py
python multexp_seg.py
python multexp_seg_scale.py
python multexp_thr.py
python multexp_transfer.py
```

# Example
Running the segmentation experiment will generate csv files for the data and the plots shown below:

![alt text](https://github.com/ivadomed-profile-analysis-project/ivadomed-profile-analysis/blob/main/experiments/plots/seg_1xdata/subplot.png)
![alt text](https://github.com/ivadomed-profile-analysis-project/ivadomed-profile-analysis/blob/main/experiments/plots/seg_1xdata/time_per_comp_plot.png)
![alt text](https://github.com/ivadomed-profile-analysis-project/ivadomed-profile-analysis/blob/main/experiments/plots/seg_1xdata/training_subplot.png)
![alt text](https://github.com/ivadomed-profile-analysis-project/ivadomed-profile-analysis/blob/main/experiments/plots/seg_1xdata/validation_subplot.png)
