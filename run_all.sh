#!/bin/bash

# Quick Start Script for RSA DQN Project
# This script sets up and runs the complete training and evaluation pipeline

echo "=========================================="
echo "RSA DQN Project - Quick Start"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo ""
echo "Installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Test environment
echo ""
echo "=========================================="
echo "Testing Environment"
echo "=========================================="
python test_env.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Environment test failed! Please check the errors above."
    exit 1
fi

# Ask user if they want to proceed with training
echo ""
read -p "Environment test passed! Proceed with training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled. You can run 'python dqn_runner.py' later to train."
    exit 0
fi

# Train models
echo ""
echo "=========================================="
echo "Training DQN Models"
echo "=========================================="
echo "This will take approximately 30-60 minutes..."
python dqn_runner.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Training failed! Please check the errors above."
    exit 1
fi

# Ask user if they want to proceed with evaluation
echo ""
read -p "Training complete! Proceed with evaluation? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Evaluation cancelled. You can run 'python evaluate.py' later."
    exit 0
fi

# Evaluate models
echo ""
echo "=========================================="
echo "Evaluating Trained Models"
echo "=========================================="
python evaluate.py

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "Results are available in:"
echo "  - models/     : Trained model files (.zip)"
echo "  - plots/      : All generated plots (.png)"
echo "  - results/    : Evaluation metrics (.json)"
echo ""
echo "See README.md for detailed documentation."
