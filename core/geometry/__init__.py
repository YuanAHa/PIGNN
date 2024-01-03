from fenics import Point
from mshr import generate_mesh, Rectangle
import numpy as np
from enum import IntEnum
import torch_geometric.transforms as T
from torch_geometric.data import Data
import torch

class NodeType(IntEnum):
    inner=0
    boundary=1

def get_node_type(pos, radius_ratio=None):
    max_x = np.max(pos[:, 0])
    max_y = np.max(pos[:, 1])
    min_x = np.min(pos[:, 0])
    min_y = np.min(pos[:, 1])
    
    right = np.isclose(pos[:, 0], max_x)
    left = np.isclose(pos[:, 0], min_x)
    up = np.isclose(pos[:, 1], max_y)
    bottom = np.isclose(pos[:, 1], min_y)    
    
    on_boundary = np.logical_or(np.logical_or(right, left),np.logical_or(up, bottom))
    
    node_type = np.ones((pos.shape[0], 1))
    node_type[on_boundary] = NodeType.boundary
    node_type[np.logical_not(on_boundary)] = NodeType.inner
        
    return np.squeeze(node_type)
    

class RectangleMesh():
    
    node_type_ref = NodeType
    def __init__(self, density=100, lb=(0, 0), ru=(1, 1)) -> None:
        
        self.transform = T.Compose([
            T.FaceToEdge(remove_faces=False), 
            T.Cartesian(norm=False), 
            T.Distance(norm=False)
            ])
        domain = Rectangle(Point(lb[0],lb[1]), Point(ru[0], ru[1]))
        self.mesh = generate_mesh(domain, density)
        self.pos = self.mesh.coordinates().astype(np.float32)
        self.faces = self.mesh.cells().astype(np.int64).T        
        self.node_type = get_node_type(self.pos).astype(np.int64)
        print("Node numbers: %d"%self.pos.shape[0])
        
    def getGraphData(self):
        graph = Data(pos=torch.as_tensor(self.pos), 
                    face=torch.as_tensor(self.faces))
        graph = self.transform(graph)
        graph.num_nodes = graph.pos.shape[0]
        graph.node_type = torch.as_tensor(self.node_type)
        graph.label = 0
        return graph

