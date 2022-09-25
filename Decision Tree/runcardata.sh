#!/bin/sh
echo "Results for car dataset"
python3 DecisionTree.py

echo "Results for bank dataset - unknown as feature value"
python3 DecisionTree_bank.py

echo "Results for bank dataset - unknown as missing"
python3 DecisionTree_bank_missing.py
