#!/bin/sh
echo "Results for SVM ALGORITHM"
echo "Results for Primal SVM"
python3 Primal_SVM.py

echo "Results for Dual SVM"
python3 Dual_SVM.py

echo "Results for Kernel Perceptron"
python3 kernel_perceptron.py