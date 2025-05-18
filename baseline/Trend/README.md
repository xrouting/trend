### How to Run

✅ **Please remember to modify the port number and file paths in the code as needed to match your local environment.**

#### 1. Train the Transformer-based RL Agent

Run the following command to start training:

Server-side Startup

````
cd	xroute_env/baseline/Trend/openroad_api/demo/xroute_simulation_platform
python trainer_auto_switch_2.py

````

Client-side Startup

```
python train_trend.py 
```

#### 2. Evaluate the Trained Model

```
./inference.sh
```


