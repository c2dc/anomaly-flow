# Anomaly-Flow

Framework for implementing Intrusion Detection Systems (NIDS) aimed at identifying anomalies in network flows using Machine Learning models.

### Create the Experimental Environment

To create the experimental environment, execute the following commands: 

```sh
python -m venv .env 
```

**Note: The virtual environment must have this name because the scripts use this name to load the needed libraries.**

After creating the Virtual environment, use the following command to install the necessary python packages: 

```sh
pip install -r requirements.txt
```

### Commands to run local experiments 

To run the local experiments, use the following command: 

```sh
source experiments.sh
```

### Commands to run the Federated Learning experiments

To run the Federated Learning experiments, use the following command: 

```sh
source run.sh 
```

### Getting the data to run the experiments

To run the experiments, we need to have the CIC-IDS2018, BoT-IoT and ToN-IoT in a folder called datasets in the root of the project, e.g. : 

    |- anomaly-flow
        |- datasets 
            |- NF-CSE-CIC-IDS2018-v2-DDoS.csv.gz
            |- NF-BoT-IoT-v2-DDoS.csv.gz
            |- NF-ToN-IoT-v2-DDoS.csv.gz 

In our case we used derived datasets containing only benign and DDoS samples filtered from the originals datasets. 


### Generating Synthetic Data

To generate Synthetic Data for a specific dataset use the script [main.ipynb](https://github.com/leonardohdemelo/anomaly-flow/blob/main/main.ipynb). 

### External models Simple Models Baseline

#### Install the EFC package 

To run the external and simple model scripts, please install the custom pip package for the EFC algorithm:

1. Make sure the scripts have run privileges: 

```sh
chmod +x ./auxiliary_scripts/install-efc.sh
```

2. Run the script (Git and the anomaly-flow environment named **.env** required):

```sh
./auxiliary_scripts/install-efc.sh
```

## References 

The data used to carry out the experiments can be obtained from:

[Machine Learning-Based NIDS Datasets (Netflow V2 Datasets)](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) 

The following projects were used as reference for the preparation of this work:

[GANomaly](https://github.com/samet-akcay/ganomaly)

[Anomaly-Toolbox Project](https://github.com/zurutech/anomaly-toolbox) 

[Energy-based Flow Classifier](https://github.com/EnergyBasedFlowClassifier/EFC-package)

## Authors 

<a href="https://github.com/leonardohdemelo"><img src="https://avatars0.githubusercontent.com/u/43916660?s=460&v=4" alt="drawing" width="40" align="middle"/></a>
&nbsp;&nbsp;&nbsp;Leonardo Henrique de Melo 

<a href="https://github.com/gubertoli"><img src="https://avatars.githubusercontent.com/u/4803756?v=4" alt="drawing" width="40" align="middle"/></a>
&nbsp;&nbsp;&nbsp;Gustavo de Carvalho Bertoli

<a href="https://homepages.dcc.ufmg.br/~michele/"><img src="https://servicosweb.cnpq.br/wspessoa/servletrecuperafoto?tipo=1&id=K4701867Y5" alt="drawing" width="40" align="middle"/></a>
&nbsp;&nbsp;&nbsp;Michele Nogueira

<a href="https://dcc.ufmg.br/professor/aldri-luiz-dos-santos/"><img src="https://dcc.ufmg.br/wp-content/uploads/2021/01/Aldri.jpg" alt="drawing" width="40" align="middle"/></a>
&nbsp;&nbsp;&nbsp;Aldri Luiz dos Santos

<a href="https://github.com/ljr"><img src="https://avatars.githubusercontent.com/u/978047?v=4" alt="drawing" width="40" align="middle"/></a>
&nbsp;&nbsp;&nbsp;Lourenço Alves Pereira Junior
