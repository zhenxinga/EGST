# EGST
A program for EV Charging demand
# EG-ST: Spatio-Temporal Graph Learning for Electric Vehicle Charging Demand Prediction
This repository provides the implementation of **EG-ST**, a spatio-temporal graph learning model for electric vehicle (EV) charging demand prediction. The model predicts future EV charging demand by jointly modeling temporal charging patterns, price information, and spatial relationships among regions or charging stations.
This project is developed for experiments on two EV charging demand datasets:
- **ST-EVCDP**: The dataset used by the reference model adopted in this project.  
  Dataset link: https://github.com/IntelligentSystemsLab/ST-EVCDP
- **UrbanEV**: The second dataset used in this project.  
  Download link: https://github.com/IntelligentSystemsLab/UrbanEV

## Project Structure
EG-ST/
  main.py                    Main script for training, validation, and testing
  model.py                   EG-ST model implementation
  learner.py                 Physics-informed pre-training and fast learning module
  functions.py               Data loading, dataset construction, and evaluation metrics
  baselines.py               Baseline model implementations
LICENSE                    License file
checkpoints/               Directory for saving trained model checkpoints
results/                   Directory for saving testing results
datasets/                  Dataset directory prepared by users
## Model Overview
EG-ST is designed to capture both temporal dependencies and spatial correlations in EV charging demand. The model mainly consists of the following components.
### Local Temporal Encoding
The model uses convolutional layers to extract local temporal features from historical charging occupancy and charging price sequences.
### Temporal Embedding Fusion
Node-level temporal embeddings are introduced to enhance the temporal representation of charging regions or stations. In this version of the project, the temporal feature representation and the temporal embedding representation are fused:
Here, h0 denotes the temporal feature representation, and sp denotes the spatial embedding representation.

### Static Graph Convolution
The static graph convolution module uses the adjacency matrix to model fixed spatial relationships among regions or stations.
### Dynamic Graph Construction
A dynamic graph is constructed according to historical charging occupancy patterns. This allows the model to capture time-varying correlations in EV charging demand.
### Attention-based Temporal Modeling
The fused spatio-temporal features are further processed by a attention module to generate final charging demand predictions.

## Requirements
Python 3.8 or later is recommended.
The main dependencies are:
torch
numpy
pandas
scikit-learn
tqdm
You can install them with:
pip install torch numpy pandas scikit-learn tqdm
If GPU acceleration is used, please install the PyTorch version that matches your CUDA environment.
## Dataset Preparation
By default, the code reads data from the datasets/ directory under the project root.
Expected file structure:
EG-ST/
datasets/
  occupancy.csv
  price.csv
  adj.csv
  distance.csv
  information.csv
  time.csv
The expected files are:
occupancy.csv: Charging occupancy or charging demand records over time.
price.csv: Charging price records over time.
adj.csv: Adjacency matrix describing spatial connections among regions or stations.
distance.csv: Distance matrix among regions or stations.
information.csv: Basic node information, such as charging capacity or pile numbers.
time.csv: Timestamp sequence.
## ST-EVCDP Dataset
ST-EVCDP is the dataset used by the reference model adopted in this project. It can be downloaded from:
https://github.com/IntelligentSystemsLab/ST-EVCDP
After downloading the dataset, please organize the required CSV files according to the input format expected by this project and place them under the datasets/ directory.
## UrbanEV Dataset
UrbanEV is the second dataset used in this project. It can be downloaded from:
https://github.com/IntelligentSystemsLab/UrbanEV
The original UrbanEV file names and fields may be different from the default input format of this project. Before running the code, please preprocess UrbanEV into the format required by the read_dataset() function in functions.py.
If both datasets are kept in the same project, the following structure is recommended:
EG-ST/
datasets/
  ST-EVCDP/
  UrbanEV/
Then modify the data loading path in functions.py accordingly.
## Running the Code
After preparing the dataset, modify the read_dataset() function in functions.py according to the dataset path and file format.
Then run the following command in the project root directory:
python main.py
The program will automatically load the dataset through read_dataset(), split the data, train and validate the model, evaluate the model on the testing set, and save the testing results into the results/ directory.

## Main Parameters
The following parameters can be modified in main.py:
seq_l: Historical input sequence length. Default value: 12.
pre_l: Prediction horizon. Default value: 6.
bs: Batch size. Default value: 512.
n_epoch: Maximum number of training epochs. Default value: 300.
patience: Early stopping patience. Default value: 10.
is_train: Whether to train the model. Default value: True.
is_pre_train: Whether to use the pre-training stage. Default value: False.
mode: Pre-training mode. Available options include simplified and completed.
The model configuration dictionary can be used to adjust the model structure, including convolution kernel size, spatial embedding dimension, time embedding dimension, dynamic embedding dimension, and dynamic graph edge dropout ratio.

## Output Files
During training, the best model is saved to:
checkpoints/egst_6_bs512_completed.pt
Testing metrics are saved to:
results/egst_6bs512.csv
The output metrics include:
MSE: Mean Squared Error.
RMSE: Root Mean Squared Error.
MAPE: Mean Absolute Percentage Error.
RAE: Relative Absolute Error.
MAE: Mean Absolute Error.
R2: Coefficient of Determination.
## Notes
Before running the code, make sure that the datasets/ directory exists and contains all required CSV files.
If UrbanEV is used, please preprocess its fields and file names to match the input format required by read_dataset().
The code uses GPU by default when CUDA is available. To run the code on CPU only, set use_cuda to False.
The spatial feature fusion in this version uses:
h = torch.cat([h0, sp], dim=-1)
If seq_l, spatial_emb_dim, or the feature fusion strategy is changed, please check the input dimensions of the following GCN, Transformer, and prediction layers accordingly.
## Dataset References
If you use the ST-EVCDP or UrbanEV datasets, please cite the original dataset repositories or the corresponding papers.
ST-EVCDP: https://github.com/IntelligentSystemsLab/ST-EVCDP
UrbanEV: https://github.com/IntelligentSystemsLab/UrbanEV
Related paper:
@Article{qu2024a,
  author={Qu, Haohao and Kuang, Haoxuan and Wang, Qiuxuan and Li, Jun and You, Linlin},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={A Physics-Informed and Attention-Based Graph Learning Approach for Regional Electric Vehicle Charging Demand Prediction}, 
  year={2024},
  pages={1-14},
  doi={10.1109/TITS.2024.3401850}}

@article{kuang2024unravelling,
  title={Unravelling the effect of electricity price on electric vehicle charging behavior: A case study in Shenzhen, China},
  author={Kuang, Haoxuan and Zhang, Xinyu and Qu, Haohao and You, Linlin and Zhu, Rui and Li, Jun},
  journal={Sustainable Cities and Society},
  pages={105836},
  year={2024},
  publisher={Elsevier}
}

@article{qu2024chatev,
 title = {ChatEV: Predicting electric vehicle charging demand as natural language processing},
 journal = {Transportation Research Part D: Transport and Environment},
 volume = {136},
 pages = {104470},
 year = {2024},
 issn = {1361-9209},
 author = {Haohao Qu and Han Li and Linlin You and Rui Zhu and Jinyue Yan and Paolo Santi and Carlo Ratti and Chau Yuen},
}

@article{li2025urbanev,
 title = {UrbanEV: An Open Benchmark Dataset for Urban Electric Vehicle Charging Demand Prediction},
 journal = {Scientific Data},
 volume = {12},
 pages = {523},
 year = {2025},
 issn = {2052-4463},
 author = {Li, Han and Qu, Haohao and Tan, Xiaojun and You, Linlin and Zhu, Rui and Fan, Wenqi},
}
## License

This project keeps the original LICENSE file. When using the code and datasets, please follow the license requirements of both this project and the original dataset repositories.
