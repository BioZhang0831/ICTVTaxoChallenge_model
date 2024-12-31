# ICTVTaxoChallenge_model
   This is a deep learning model developed to participate in the challenge proposed by ICTV. Its primary function is to classify a set of unknown viral sequences. This model is based on GP-GCN (Gapped Pattern Graph Convolutional Networks) that utilizes RSCU (Relative Synonymous Codon Usage) and TNF (Tetra-Nucleotide Frequencies) to improve the annotation accuracy of the  taxonomic classifier. The model is trained on annotated viruses dataset published by ICTV.
   
   Our results can be found in the result folder, with the file named as `ICTVTaxoChallenge_taxonomy_result.tsv`

   All dataset and features used for model training and testing can be found in the Baidu disk link file `dataprocess/Baidu_disk_link`.
# Getting started 
## Prerequisite
* cython
* numpy
* biopython
* editdistance
* pytorch 1.7.1
* pytorch_geometric 1.7.0

## Install by Conda

```shell
conda env create -f environment.yml
```

## Code Replication Steps
0. Preprocess(`dataprocess/GCN_embedding_pipline.ipynb dataprocess/calculate_RSCU.py dataprocess/calculate_TNF.py`)
This Jupyter notebook includes the embedding process using GP-GCN (Gapped Pattern Graph Convolutional Networks) to embed the data. First, a model is pre-trained using the GCNmodel, and then the input data is processed to generate the embedded matrix. The scripts calculate_RSCU.py and calculate_TNF.py are used to generate the feature matrices.
2. Data Processing (`dataprocess/process.ipynb`)
This Jupyter notebook contains the initial steps required to process the raw data. It includes data cleaning, normalization, and splitting the data into training and testing datasets. Ensure you have all the necessary libraries installed and that the raw data is located in the specified directory before running this notebook.
3. Model Training (`model/train.ipynb`)
In this notebook, the machine learning model is trained using the processed data. This includes the selection of model parameters, training the model, and evaluating its performance on the training dataset.
4. Prediction Data Processing (`dataprocess/prediction_process.ipynb`)
This notebook prepares the data that will be used for making predictions. It processes new or unseen data similar to how the training data was processed, ensuring consistency in data handling.
5. Model Prediction (`model/prediction.ipynb`)
Use the trained model to make predictions on new data processed in the previous step. This notebook loads the trained model, inputs the processed data, and outputs the predictions.
6. Results Processing (`dataprocess/result_process.ipynb`)
After obtaining predictions, this notebook is used to process the results. 
