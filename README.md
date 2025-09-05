# PLPPI

###
Code repo for paper: [Sensorless End-to-End Freehand Three-dimensional Ultrasound Reconstruction with Physics Guided Deep Learning](https://ieeexplore.ieee.org/document/10684746). 

Fifth solution at MICCAI TUS-REC 2024.

We adapted the framework from [challenge baseline](https://github.com/QiLi111/tus-rec-challenge_baseline). 
## Training Code

### Instruction
This repository provides an example framework for freehand US pose regression from [challenge baseline](https://github.com/QiLi111/tus-rec-challenge_baseline), including usage of various types of predictions and labels (see [transformation.py](https://github.com/Alphafrey946/PLPPI/blob/main/utils/transform.py)). Please note that the networks used here are small and simplified for demonstration purposes.

For instance, the network can predict the transformation between two US frames as 6 DOF "parameter". If the label type is "point", the loss is calculated as the point distance (by transforming "parameter" to "point" using function [parameter_to_point](https://github.com/Alphafrey946/PLPPI/blob/main/utils/transform.py#L267)). The steps below illustrate an example of training a pose regression model and generate 4 kinds of displacements. 

<!-- We use the transformation from image coordinate system (in mm) to image coordinate system (in mm), for example described in function [to_transform_t2t](https://github.com/Alphafrey946/PLPPI/blob/main/utils/transform.py#L93).  -->

### Steps to run the code
#### 1. Clone the repository.
```
git clone https://github.com/Alphafrey946/PLPPI.git
```

#### 2. Navigate to the root directory.
```
cd PLPPI
```

#### 3. Install conda environment

``` bash
conda create -n freehand-US python=3.9.13
conda activate freehand-US
pip install -r requirements.txt
conda install pytorch3d --no-deps -c pytorch3d
pip install spatial-correlation-sampler --no-build-isolation
```

#### 4. Create directories.
```
mkdir -p data/frames_transfs
mkdir -p data/landmarks
```

#### 5. For Custom data
Follow the format of the of the [challenge](https://zenodo.org/records/11178509)

#### 6. Make sure the data folder structure is the same as follows.
```bash
├── data/ # Contains data set 
│ ├── frames_transfs/ 
│  ├── 000/ # scans in subject 000
│    ├── **.h5 # Scan info including frames and transformations 
│    ├── ...
│  ├── ...
```

#### 7. Train model. 
``` bash
python3 train.py
```
#### 8. Generate DDF.
``` bash
python3 generate_DDF.py
```


* Challenge paper:
  * Qi Li et al. "TUS-REC2024: A Challenge to Reconstruct 3D Freehand Ultrasound Without External Tracker." arXiv preprint arXiv:<a href="https://doi.org/10.48550/arXiv.2506.21765" target="_blank">2506.21765</a>(2025). 