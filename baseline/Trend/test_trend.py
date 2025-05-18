import torch
import zmq
from utils import handle_messange,Game,check_pass
import openroad_api.proto.net_ordering_pb2 as net_ordering
import time
import json
from discrete_A3C import Net
from torch_geometric.data import Data
from models.actor_critic import Actor, Critic


def load_model(ckp_dir,actor,critic,optimizer): 
    checkpoint = torch.load(ckp_dir)
    actor.load_state_dict(checkpoint['actor_state_dict'])
    critic.load_state_dict(checkpoint['critic_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return actor,critic,optimizer

class InferenceSelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, checkpoint_path=None,port='6666',device='cpu'):
        #self.model.set_weights(initial_checkpoint["weights"])
        #加载环境和模型   
        self.game = Game()
        self.actor = Actor(device)
        self.critic = Critic(device)
        self.optimizer = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=0.00005, eps=1e-5)
        self.actor,self.critic,self.optimizer = load_model(checkpoint_path,self.actor,self.critic,self.optimizer)
        self.port = port
        self.idx = 1
        self.unpass_count = 0

    def response_action(self):
        self.actor.eval()
        self.critic.eval()
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:"+self.port)
        print('Server started.')
        while True:
            message_raw = socket.recv()
            print("receive")
            start_time = time.time()
            #解析消息
            message = net_ordering.Message()
            message.ParseFromString(message_raw)
            data = handle_messange(message,socket)
            #print("spend time:",time.time()-start_time)
            s = self.game.get_feature(data)
            pass_flag = check_pass(s)
            #print("spend time:",time.time()-start_time)
            #print("s:",s)
            n = len(s['graph_node_properties'])
            action_list = [ i for i in range(n)]
            if pass_flag == False:
                #self.unpass_count += 1
                x = torch.tensor(s['graph_node_properties'], dtype=torch.float).to(device)
                edge_index = torch.tensor(s['graph_edge_connections'], dtype=torch.long).to(device)
                if len(edge_index):
                    edge_index = torch.cat((edge_index, edge_index[:, [1, 0]]), dim=0).T
                else:
                    edge_index = torch.tensor([[], []], dtype=torch.long).to(device)
                observation = Data(x=x, edge_index=edge_index)
                with torch.no_grad():
                    outputs, _ = self.actor(observation, True)
                action_list = list(outputs.cpu().detach().numpy().flatten())
                action_list = [int(action) for action in action_list]
            message = net_ordering.Message() 
            message.response.net_list.extend(action_list)
            print(f'第 {self.idx} 次 发送 action_list = {action_list}')
            self.idx += 1
            print("spend time:",time.time()-start_time)
            socket.send(message.SerializeToString())             

if __name__ == '__main__':
    import sys
    port = sys.argv[1]
    device = torch.device("cuda:0")
    pretrained_path = 'path/to/your/model/checkpoint.pth'
    print(f'port:{port},device:{device}')
    server = InferenceSelfPlay(checkpoint_path=pretrained_path,port=port,device=device)
    server.response_action()  

# s = {'graph_node_properties': [[4, 3.0, 0.0, 0.4117647058823529, 0.34210526315789475, 0.3333333333333333, 0.19607843137254902, 0.32894736842105265, 0.0, 8], [2, 1.0, 0.0, 0.21568627450980393, 0.17105263157894737, 0.2222222222222222, 0.4117647058823529, 0.4473684210526316, 0.0, 5], [2, 2.5, 0.0, 0.29411764705882354, 0.15789473684210525, 0.2222222222222222, 0.29411764705882354, 0.4605263157894737, 0.0, 3], [3, 1.6666666666666667, 0.0, 0.3137254901960784, 0.013157894736842105, 0.3333333333333333, 0.39215686274509803, 0.5526315789473685, 0.0, 1], [3, 2.0, 0.0, 0.3137254901960784, 0.32894736842105265, 0.3333333333333333, 0.39215686274509803, 0.2894736842105263, 0.0, 6], [3, 2.3333333333333335, 0.0, 0.23529411764705882, 0.2631578947368421, 0.3333333333333333, 0.27450980392156865, 0.39473684210526316, 0.0, 5], [2, 1.0, 0.0, 0.23529411764705882, 0.02631578947368421, 0.2222222222222222, 0.17647058823529413, 0.4605263157894737, 0.1111111111111111, 1], [2, 1.0, 0.0, 0.11764705882352941, 0.039473684210526314, 0.1111111111111111, 0.1568627450980392, 0.5131578947368421, 0.1111111111111111, 0], [2, 1.0, 0.0, 0.6274509803921569, 0.013157894736842105, 0.1111111111111111, 0.13725490196078433, 0.5526315789473685, 0.1111111111111111, 0], [2, 1.0, 0.0, 0.4117647058823529, 0.02631578947368421, 0.1111111111111111, 0.19607843137254902, 0.6447368421052632, 0.1111111111111111, 0], [2, 1.0, 0.0, 0.37254901960784315, 0.13157894736842105, 0.2222222222222222, 0.39215686274509803, 0.5526315789473685, 0.2222222222222222, 4], [2, 1.0, 0.0, 0.4117647058823529, 0.02631578947368421, 0.1111111111111111, 0.29411764705882354, 0.4473684210526316, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.6274509803921569, 0.013157894736842105, 0.1111111111111111, 0.17647058823529413, 0.47368421052631576, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.6274509803921569, 0.013157894736842105, 0.1111111111111111, 0.1568627450980392, 0.5131578947368421, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.4117647058823529, 0.02631578947368421, 0.1111111111111111, 0.2549019607843137, 0.5263157894736842, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.13725490196078433, 0.02631578947368421, 0.1111111111111111, 0.6078431372549019, 0.5921052631578947, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.13725490196078433, 0.02631578947368421, 0.1111111111111111, 0.5882352941176471, 0.631578947368421, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.6274509803921569, 0.013157894736842105, 0.1111111111111111, 0.0784313725490196, 0.6710526315789473, 0.3333333333333333, 0], [2, 1.0, 0.0, 0.3137254901960784, 0.013157894736842105, 0.1111111111111111, 0.39215686274509803, 0.3026315789473684, 0.4444444444444444, 0], [2, 1.0, 0.0, 0.3137254901960784, 0.013157894736842105, 0.1111111111111111, 0.39215686274509803, 0.4342105263157895, 0.4444444444444444, 0], [2, 1.0, 0.0, 0.3137254901960784, 0.013157894736842105, 0.1111111111111111, 0.39215686274509803, 0.4605263157894737, 0.4444444444444444, 0]], 'graph_edge_connections': [[21, 17], [21, 15], [21, 9], [21, 20], [21, 19], [21, 18], [21, 14], [21, 10], [17, 15], [17, 9], [17, 20], [17, 19], [15, 9], [15, 20], [15, 19], [9, 20], [20, 19], [20, 18], [19, 9], [19, 18], [18, 15], [16, 9], [16, 11], [16, 13], [16, 2]]}
# print(len(s['graph_node_properties']),len(s['graph_node_properties'][0]))
# actor = Actor(device)
# critic = Critic(device)
# optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=0.00005, eps=1e-5)
# actor,critic,optimizer = load_model(pretrained_path,actor,critic,optimizer)
# x = torch.tensor(s['graph_node_properties'], dtype=torch.float).to(device)
# edge_index = torch.tensor(s['graph_edge_connections'], dtype=torch.long).to(device)
# if len(edge_index):
#     edge_index = torch.cat((edge_index, edge_index[:, [1, 0]]), dim=0).T
# else:
#     edge_index = torch.tensor([[], []], dtype=torch.long).to(device)
# observation = Data(x=x, edge_index=edge_index)
# with torch.no_grad():
#     outputs, _ = actor(observation, True)
# action_list = list(outputs.cpu().detach().numpy().flatten())

