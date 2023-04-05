# Fraud Detection using PyTorch

## Aim
* To implement a credit card fraud detection classifier using a neural network in PyTorch Framework

## Approach
* Basic EDA + Preprocessing
    - Checks for normality
    - Checks for pairwise correlation
    - Identification and handling of class imbalance using SMOTE (Synthetic Minority - Over-sampling Technique - https://arxiv.org/abs/1106.1813v1)
* Building a neural network using PyTorch
    - Architecture: 
        - Fully connected layer 1: Linear(in_features=30, out_features=64, bias=True)
        - Fully connected layer 2: Linear(in_features=64, out_features=32, bias=True)
        - Fully connected layer 3: Linear(in_features=32, out_features=16, bias=True)
        - Fully connected layer 4: Linear(in_features=16, out_features=1, bias=True)
    - With a batch size of 64 and over 5 epochs, the NN was trained with Cross Entropy as the loss and 0.1 learning rate.

## Conclusion
The model seems to give similar training and test accuracies. The variation in loss & accuracy across epochs is very small and hence we might need to increase the number of epochs and/or try a different neural network architecture for improved performance.



## Data
The dataset contains transactions made by credit cards in September 2013 by European cardholders. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
For more details, see https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud. 