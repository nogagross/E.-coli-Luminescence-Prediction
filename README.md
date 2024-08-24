# E.-coli-Luminescence-Prediction

## Overview
This project involves the prediction of E. coli luminescence in the presence and absence of DNT using machine learning models. We utilized **XGBoost** and **Random Forest** models to predict the outcomes based on various features derived from DNA promoter sequences.

## Features
The features used in this project include:

- **Mean and Maximum Values**: Calculated for certain sequences to derive useful insights.
- **Nucleotide Changes**: Tracks the number of nucleotide changes in comparison to a control variant, helping to identify genetic patterns.
- **Pairwise Nucleotide Impact**: Considers the impact of all nucleotide pairs, aiding in understanding the relationships between genetic sequences and their effects.
- **Melting Temperature**: Measures the temperature at which DNA strands separate, serving as an indicator of nucleotide pair stability.
- **Robustness**: Assesses the model's ability to handle data variability and noise.

Additional features derived and used for model training include:
- **Folding Energy**: Computed across windows of 13 nucleotides and averaged across every 20 windows to represent local stability.
- **Codon Changes**: Tracks the number of codon changes in the variant compared to the control.
- **TA Dinucleotide Count**: Counts the occurrences of TA dinucleotides, which are crucial for certain genetic interactions.
- **PSSM Scores**: Maximum PSSM scores for specific motifs and the index where the score is found, offering insights into sequence alignment and potential biological significance.

## Models
- **XGBoost**: A powerful gradient boosting framework that combines multiple weak models to create a strong predictive model. It is particularly suited for handling large datasets and complex relationships within the data.
- **Random Forest**: An ensemble learning method that builds multiple decision trees and merges them to obtain more accurate and stable predictions.

## Data
The data includes various genetic sequences, features derived from DNA promoter sequences, and corresponding luminescence outcomes under different conditions. The models were trained and tested on this data to predict luminescence effectively.

## Installation
To set up the project on your local machine, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/your-repository.git
    ```
2. Navigate to the project directory:
    ```sh
    cd your-repository
    ```
3. Install the necessary dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To run the model and make predictions:

1. Ensure the dataset is available in the `data` directory.
2. Run the following command:
    ```sh
    python model.py
    ```

## Results
The models were evaluated using various metrics, including Spearman's correlation, to ensure the predictions reliability. Notably, the **XGBoost** model outperformed **Random Forest** in this task, achieving higher correlations.

## License
This project is licensed under the MIT License. 
