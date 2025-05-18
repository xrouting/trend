# Trend

This repository provides the official implementation of the paper: **Transformer-based Reinforcement Learning for Net Ordering in Detailed Routing**

### Paper Summary

In modern VLSI design, detailed routing becomes increasingly challenging as feature sizes shrink and design complexity grows. One key factor affecting routing quality is the ordering of nets, especially in sequential routers using rip-up and reroute strategies. Traditional net ordering relies heavily on hand-crafted heuristics.

In this work, we propose a Transformer-based reinforcement learning framework to learn net ordering policies directly from routing experience (success/failure), aiming to improve routing quality automatically. Our related paper, **Transformer-based Reinforcement Learning for Net Ordering in Detailed Routing**, has been accepted at IJCAI-2025.

## Quickstart

This code relies on the xroute_env environment for simulating the detailed routing task.

### Installation

To interact with the xroute environment, you need to download the simulator first:

| Operating System | Download Link                                                |
| ---------------- | ------------------------------------------------------------ |
| Ubuntu 22.04     | [Download](https://drive.google.com/drive/folders/1cFmv9EXe319hs_NlaoN_ZHRLRlXw7Zks?usp=sharing) |

Then, put the simulator in the `third_party/openroad` folder.

You may also need to execute the following command to install some libraries to ensure that OpenRoad can start up properly.

```
cd third_party/openroad
chmod +x DependencyInstaller.sh
source ./DependencyInstaller.sh
```

### Agent Introduction

[Trend](https://github.com/xrouting/trend/blob/main/baseline/Trend/README.md)

[A3C](https://github.com/xrouting/trend/blob/main/baseline/A3C/README.md)

### Launch Mode

You can choose to launch the simulator in following modes:

#### Training Mode

In this mode, the simulator should launch first, then the agent can control the simulator to train the model.

```
cd examples && python3 launch_training.py

cd baseline/Trend && python train_trend.py 
# cd baseline/A3C && python discrete_A3C.py
```

After executing the command above, the simulator will listen to the port 6667 to wait for environment reset command, and then interact with the agent via port 5556.

#### Evaluation Mode

In this mode, the agent should launch first, then the simulator can connect to the agent to get the action.

```
cd baseline/Trend && ./inference.sh
# cd baseline/A3C && python3 test_A3C.py 5556 cpu
```

### Acknowledgement

The routing simulator in xroute environment is mainly based on [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) TritonRoute. Thanks for their wonderful work!









