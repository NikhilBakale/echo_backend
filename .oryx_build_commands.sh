#!/bin/bash

# Pre-build script for Azure Oryx
echo "Running pre-build setup..."

# Ensure pip is upgraded
python -m pip install --upgrade pip setuptools wheel

# Set pip configuration for better compatibility
pip config set global.timeout 1000
pip config set global.retries 5

echo "Pre-build setup completed"
