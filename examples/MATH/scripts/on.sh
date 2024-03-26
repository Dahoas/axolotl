#!/bin/bash

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script
H=`hostname`
RANK=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

cd /weka/home-alex/repos/axolotl/examples/MATH/
source /weka/home-alex/.envs/ft/bin/activate

echo $MASTER_ADDR
echo $MASTER_PORT

accelerate launch --num_processes $((8 * $COUNT_NODE)) --num_machines $COUNT_NODE --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
--config_file configs/accelerate_configs/train_z2.yaml -m axolotl.cli.train configs/axolotl_configs/math.yaml
