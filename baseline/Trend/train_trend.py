import numpy as np
import torch
import pathlib
import os
import datetime
import time
import math
from models.actor_critic import Actor, Critic
import argparse
from utils import Game
import torch
from torch_geometric.data import Data



def load_model(ckp_dir,actor,critic,optimizer): 
    checkpoint = torch.load(ckp_dir)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('load model from',ckp_dir)
    return actor,critic,optimizer

class Worker:
    def __init__(self,results_path, total_step,server_port='6666',client_port='5555',is_load = True,load_dir=None,learning_rate=0.00005):
        self.total_step = total_step
        self.env = Game(server_port=server_port,client_port=client_port)
        self.results_path = results_path
        self.n_per_layout = 10
        self.actor = Actor(device)
        self.critic = Critic(device)
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=learning_rate, eps=1e-5)
        if is_load and os.path.exists(load_dir):
            self.actor,self.critic,self.optimizer = load_model(load_dir,self.actor,self.critic,self.optimizer)


    def run(self):
        start_time = time.time()
        self.actor.train()
        self.critic.train()
        train_step = 1            
        #启动训练时把布局调整到第一个
        bool_reset = True
        bool_jump = False
        print('start training')
        s = {}
        while train_step < self.total_step:
            #训练交互前重置为第一个布局，每交互完n_per_layout个布局则跳转下一个布局
            #空布局一律跳过
            #若s为空，则先reset，若还为空则还在循环，执行带jump的reset
            #若s不空，最开始reset为True，后续都为False，可以保证后续都jump。
            #正常执行一轮最后jump一定是true。只有s不空且不jump才能进入训练。
            while s == {} or bool_jump:
                if bool_reset:
                    s = self.env.reset(bool_reset=True)
                    bool_reset = False
                else:
                    s = self.env.reset(bool_jump=True)
                if s != {} and bool_jump:
                    bool_jump = False
                    break

            while True:
                #handle observation
                x = torch.tensor(s['graph_node_properties'], dtype=torch.float).to(device)
                edge_index = torch.tensor(s['graph_edge_connections'], dtype=torch.long).to(device)
                if len(edge_index):
                    edge_index = torch.cat((edge_index, edge_index[:, [1, 0]]), dim=0).T
                else:
                    edge_index = torch.tensor([[], []], dtype=torch.long).to(device)
                observation = Data(x=x, edge_index=edge_index)
                print("observation:",observation.x.shape)
                net_list,log_probs = self.actor(observation)
                predictions = self.critic(observation)
                print("predictions:",predictions)
                net_list = net_list.tolist()[0]
                r,done,next_s = self.env.step(net_list,train_step)
                dr = -r
    
                with torch.no_grad():
                    disadvantage = dr - predictions
                actor_loss = torch.mean(disadvantage * log_probs)
                critic_loss = disadvantage.pow(2).mean()
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
                self.optimizer.step()
                
                s = next_s

                if train_step % self.n_per_layout == 0 or done:  
                    print('into break loop')
                    print("spend_time:",time.time()-start_time)
                    #save model
                    torch.save({
                        'actor_state_dict': self.actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                    },self.results_path)
                    print('results saved at',self.results_path)
                    #切换区域
                    self.env.send_0()  
                    bool_jump = True         
                    train_step += 1
                    break
                
                train_step += 1
                print(f'train_step = {train_step} , total_step = {self.total_step}')
                # initial 环境
                self.env.send_0()
                self.env.reset()

learning_rate = 0.00002
device = torch.device("cuda:1")
# device = torch.device("cpu")
results_path = pathlib.Path(__file__).resolve().parents[0] / "results" / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

ckp_dir = str(results_path) + '/' + 'A2C' + '.pt'
if not os.path.exists(results_path):
    os.makedirs(results_path)
    print(f'create dir:{results_path}')

worker = Worker(results_path=ckp_dir,total_step=20000,load_dir='path/to/your/model.pt')
worker.run()

