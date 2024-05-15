#!/bin/bash 

source ../.env/bin/activate
git clone https://github.com/EnergyBasedFlowClassifier/EFC-package
cd EFC-package
pip install -r requirements.txt
pip install .