#!/bin/bash

source .env/bin/activate

python main.py --train-dataset NF-CSE-CIC-IDS2018-v2-DDoS-downsample --test-datasets NF-BoT-IoT-v2-DDoS-downsample NF-ToN-IoT-v2-DDoS-downsample
python main.py --train-dataset NF-BoT-IoT-v2-DDoS-downsample --test-datasets NF-CSE-CIC-IDS2018-v2-DDoS-downsample NF-ToN-IoT-v2-DDoS-downsample
python main.py --train-dataset NF-ToN-IoT-v2-DDoS-downsample --test-datasets NF-CSE-CIC-IDS2018-v2-DDoS-downsample NF-BoT-IoT-v2-DDoS-downsample 