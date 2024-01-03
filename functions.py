import torch
from core.pde import laplacian, grad
import numpy as np


class HeatFunc(): 

    def __init__(self, delta_t, params) -> None:
        self.delta_t = delta_t
        self.params = params
        self.laplacianop = laplacian()
        self.gradop = grad()

    def graph_modify(self, graph, **argv)->None:
        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
        t = argv.get('time')
        a, b, c, d = self.params
        f = a*torch.cos(b*np.pi*(x-d)*t + c*y**2) * (b*np.pi*(x-d) - 2*c) + \
            a*torch.sin(b*np.pi*(x-d)*t + c*y**2) * (b**2 * np.pi**2 * t**2 + 4 * c**2 * y**2)
        graph.x = torch.cat((graph.x, f), dim=-1)   #节点特征添加 source_f 项  
        return graph    
        
    def init_condition(self, pos):
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        a, b, c, d = self.params
        return  a*torch.sin(c*y**2) 

    def boundary_condition(self, pos, t, **argv):
        x = pos[:, 0:1]
        y = pos[:, 1:2]
        a, b, c, d = self.params
        return  a*torch.sin(b*np.pi*(x-d)*t + c*y**2) 
    
    @classmethod
    def exact_solution(cls, pos, t):
        return cls.boundary_condition(pos, t)
    
    def pde(self, graph, values_last, values_this, **argv):

        x = graph.pos[:, 0:1]
        y = graph.pos[:, 1:2]
  
        t = argv.get('time')
        a, b, c, d = self.params
        f = a*torch.cos(b*np.pi*(x-d)*t + c*y**2) * (b*np.pi*(x-d) - 2*c) + \
            a*torch.sin(b*np.pi*(x-d)*t + c*y**2) * (b**2 * np.pi**2 * t**2 + 4 * c**2 * y**2)
        loss = (values_this - values_last)/self.delta_t - self.laplacianop(graph, values_this) - f      

        return loss.square()


class BurgesFunc():

    def __init__(self, delta_t, R) -> None:
        self.delta_t = delta_t
        self.R = R
        self.laplacianop = laplacian()
        self.gradop = grad()

    def graph_modify(cls, graph, **argv)->None:
        f = torch.zeros((graph.num_nodes, 1), device=graph.x.device)
        graph.x = torch.cat((graph.x, f), dim=-1)
        return graph
        
    def init_condition(self, pos):
        x = pos[:, 0:1]
        y = pos[:, 1:2]        
        item = self.R*(-4*x + 4*y)/32
        u = 3/4  - 1/(4*(1 + torch.exp(item)))
        v = 3/4  + 1/(4*(1 + torch.exp(item))) 
        return torch.cat((u, v), dim=-1)

    def boundary_condition(self, pos, t, **argv):
        x = pos[:, 0:1]
        y = pos[:, 1:2] 
        item = self.R*(-t -4*x + 4*y)/32
        u = 3/4  - 1/(4*(1 + torch.exp(item)))
        v = 3/4  + 1/(4*(1 + torch.exp(item))) 

        return torch.cat((u, v), dim=-1)
    
    # @classmethod
    def exact_solution(cls, pos, t):
        return cls.boundary_condition(pos, t)
    
    def pde(self, graph, values_last, values_this, **argv):
        
        u_last = values_last[:, 0:1]
        v_last = values_last[:, 1:2]

        u_this = values_this[:, 0:1]        
        v_this = values_this[:, 1:2]

        dudt = (u_this - u_last)/self.delta_t
        dvdt = (v_this - v_last)/self.delta_t

        gradvalue = self.gradop(graph, values_this)
        gradu = gradvalue[0]
        gradv = gradvalue[1]
        
        lapvalue = self.laplacianop(graph, values_this)
        lapu = lapvalue[:, 0:1]
        lapv = lapvalue[:, 1:2]

        lossu = dudt + (gradu*values_this).sum(dim=-1, keepdim=True) - 1/self.R*lapu
        lossv = dvdt + (gradv*values_this).sum(dim=-1, keepdim=True) - 1/self.R*lapv

        # return (lossu.square() + lossv.square())/2
        return torch.cat([lossu, lossv], axis=1)  


class FitzHughNagumoFunc():    
    
    def __init__(self, delta_t) -> None:        
        self.gammaru = 0.001
        self.gammarv = 0.001       
        self.alpha = 0
        self.beta = 1    
        self.delta_t = delta_t
        self.laplacianop = laplacian()

    def graph_modify(cls, graph, **argv)->None:
        f = torch.zeros((graph.num_nodes, 1), device=graph.x.device)
        graph.x = torch.cat((graph.x, f), dim=-1)
        return graph

        
    def fft(self, pos, a=1,b=1,c=1,d=0):
        pi = 2*np.pi
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos)

        x = pos[:, 0:1]
        y = pos[:, 1:2]
        center_x = (torch.max(x) + torch.min(x))/2
        center_y = (torch.max(y) + torch.min(y))/2
        y = y-center_y
        x = x-center_x
        xx = torch.cos(a*pi*(y-d))*torch.sin(b*pi*x)*torch.exp(-c*(torch.pow(x, 2) + torch.pow(y, 2)))
        yy = torch.cos(a*pi*(x-d))*torch.sin(b*pi*y)*torch.exp(-c*(torch.pow(x, 2) + torch.pow(y, 2)))
        v = torch.cat([xx, yy], dim=-1)

        return v      
        
    def init_condition(self, pos):        
        v = self.fft(pos, a=1, b=1)
        return v
    
    def boundary_condition(self, pos, t, **argv):
        if not isinstance(pos, torch.Tensor):
            pos = torch.from_numpy(pos)
        v = torch.zeros((pos.shape[0], 2), device=pos.device)
        return v
    
    def pde(self, graph, values_last, values_this, **argv):        
        u_last = values_last[:, 0:1]
        v_last = values_last[:, 1:2]

        u_this = values_this[:, 0:1]        
        v_this = values_this[:, 1:2]

        dudt = (u_this - u_last)/self.delta_t
        dvdt = (v_this - v_last)/self.delta_t
        
        lapvalue = self.laplacianop(graph, values_this)
        lapu = lapvalue[:, 0:1]
        lapv = lapvalue[:, 1:2]
        
        lossu = dudt - (self.gammaru*lapu + u_this - torch.pow(u_this, 3) - v_this + self.alpha)
        lossv = dvdt - (self.gammarv*lapv + self.beta*(u_this - v_this))

        # return (lossu.square() + lossv.square())/2
        return torch.cat([lossu, lossv], axis=1)
    
    
    

    
    
