# Conditional Entropy Calculator

This repository contains a Python script to calculate the conditional entropy of a binary label given a list of selected features from a CSV file. The script reads data
from a CSV file named data.csv, extracts the label data from column M+1, and selects the desired features based on the selected_features list. Then, it computes the
conditional entropy of the label given the selected features using the conditional_entropy function.

## Dependencies:    
numpy,     pandas

## Usage
*     Prepare your data.csv file with M features in the first M columns and the binary label data in column M+1.
*     Update the selected_features list in the script with the indices of the features you want to use for the conditional entropy calculation (e.g., [0, 1, 3] for the first, second, and fourth features).
*     Run the script using: python conditional_entropy_calculator.py

The script will output the conditional entropy H(Label|Selected Features) based on the selected features and the binary label data.

## Example

Assuming you have a data.csv file with the following content:
f1,f2,f3,f4,label
1,2,3,4,0
5,6,7,8,1
9,10,11,12,0
13,14,15,16,1
17,18,19,20,0

And you want to compute the conditional entropy of the label given the first, second, and fourth features (f1, f2, f4). Update the selected_features list in the script to [0, 1, 3] and run the script. The output will be the conditional entropy H(Label|f1, f2, f4).
