# Trend

This repository provides the official implementation of the paper:

**"Transformer-based Reinforcement Learning for Net Ordering in Detailed Routing"**

---

### Paper Summary

In modern VLSI design, detailed routing becomes increasingly challenging as feature sizes shrink and design complexity grows. One key factor affecting routing quality is the ordering of nets, especially in sequential routers using rip-up and reroute strategies. Traditional net ordering relies heavily on hand-crafted heuristics.

In this work, we propose a Transformer-based reinforcement learning framework to learn net ordering policies directly from routing experience (success/failure), aiming to improve routing quality automatically. Our related paper, **"Transformer-based Reinforcement Learning for Net Ordering in Detailed Routing," has been accepted at IJCAI 2025**.


### How to Run

✅ **Please remember to modify the port number and file paths in the code as needed to match your local environment.**

#### 1. Train the Transformer-based RL Agent

Run the following command to start training:

```
python train_A2C.py 
```

#### 2. Evaluate the Trained Model

```
./inference.sh
```


