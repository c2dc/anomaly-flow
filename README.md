# Anomaly-Flow

Framework for implementing Intrusion Detection Systems (NIDS) aimed at identifying anomalies in network flows using Machine Learning models.

### Create Experimental environment

To create the experimental environment execute the following commands: 

```sh
python -m venv .env 
```

** Note: The virtual environment must have this name, because the scripts use this name to load the needed libraries. **

After creating the Virtual environment, use the following command to install the necessary python packages: 

```sh
pip install -r requirements.txt
```

### Commands to run local experiments 

To run the local experiments use the following command: 

```sh
source experiments.sh
```

### Commands to run the Federated Learning experiments

To run the Federated Learning experiments, use the following command: 

```sh
source run.sh 
```

### Getting the data to run the experiments

To run the experiments we need to have the CIC-IDS2018, BoT-IoT and ToN-IoT in a folder called datasets in the root of the project, e.g. : 

    |- anomaly-flow
        |- datasets 
            |- NF-CSE-CIC-IDS2018-v2-DDoS.csv.gz
            |- NF-BoT-IoT-v2-DDoS.csv.gz
            |- NF-ToN-IoT-v2-DDoS.csv.gz 

In our case we used derived datasets containing only benign and DDoS samples filtered from the originals datasets. 


### Generating Synthetic Data

To generate Synthetic Data for a specific dataset use the script "main.ipynb". 

### External models Simple Models Baseline

#### Install EFC package 

To run the external and simple model scripts please install the custom pip package for the EFC algorithm:

1. Make sure the scripts has run privileges: 

```sh
chmod +x ./auxiliary_scripts/install-efc.sh
```

2. Run the script (Git and the anomaly-flow environment named **.env** required):

```sh
./auxiliary_scripts/install-efc.sh
```

## References 

[Machine Learning-Based NIDS Datasets (Netflow V2 Datasets)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) 

[Ganomaly](https://github.com/samet-akcay/ganomaly)

[Anomaly-Toolbox Project](https://github.com/zurutech/anomaly-toolbox) 

[Energy-based Flow Classifier](https://github.com/EnergyBasedFlowClassifier/EFC-package)

## Authors 

<a href="https://github.com/Ceu152"><img src="https://avatars0.githubusercontent.com/u/43916660?s=460&v=4" alt="drawing" width="40" align="middle"/></a>
Leonardo Henrique de Melo 