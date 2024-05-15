#!/bin/bash

source .env/bin/activate

# python server.py &
# sleep 60


files=("NF-CSE-CIC-IDS2018-v2-DDoS-downsample" "NF-ToN-IoT-v2-DDoS-downsample" "NF-BoT-IoT-v2-DDoS-downsample")
train_sizes=("300000" "100000" "")

# Start different anomaly flow datasets
for i in "${!files[@]}"; do
    echo "Starting client with dataset ${files[$i]}"
    python client.py --dataset=${files[$i]} --train_size=${train_sizes=[$i]} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
