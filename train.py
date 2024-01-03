import torch
from core.models import msgPassing
from core.geometry import RectangleMesh
from torch.utils.tensorboard import SummaryWriter
from core.utils.tools import parse_config, modelTrainer

device = torch.device(0)

delta_t = 1e-3

# func_name = 'heat'
# heat_params = [4. , 2. , 0.5 , 0.3]
# out_ndim = 1

func_name = 'burgers'
burgers_R = 80
out_ndim = 2

# func_name = 'fn'
# out_ndim = 2

ckptpath = 'checkpoint/simulator_%s.pth'%func_name


if func_name=='heat':
    from functions import HeatFunc as Func
elif func_name=='fn':
    from functions import FitzHughNagumoFunc as Func
elif func_name=='burgers':
    from functions import BurgesFunc as Func
else:
    raise ValueError
    

# func_main = Func(delta_t=delta_t, params=heat_params)
func_main = Func(delta_t=delta_t, R=burgers_R)
# func_main = Func(delta_t=delta_t)

ic = func_main.init_condition
bc = func_main.boundary_condition

model = msgPassing(message_passing_num=1, node_input_size=3+out_ndim, edge_input_size=3, 
                   ndim=out_ndim, device=device, model_dir=ckptpath)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

mesh = RectangleMesh(ru=(1, 1), lb=(0, 0), density=100)
graph = mesh.getGraphData().to(device)


    
train_config = parse_config()
writer = SummaryWriter('runs/%s'%func_name)   
 
setattr(train_config, 'pde', func_main.pde)
setattr(train_config, 'graph_modify', func_main.graph_modify)        
setattr(train_config, 'delta_t', delta_t)
setattr(train_config, 'ic', ic)
setattr(train_config, 'bc', bc)
setattr(train_config, 'graph', graph)
setattr(train_config, 'model', model)
setattr(train_config, 'optimizer', optimizer)
setattr(train_config, 'train_steps', 10) 
setattr(train_config, 'epchoes', 10000)
setattr(train_config, 'NodeTypesRef', RectangleMesh.node_type_ref) 
setattr(train_config, 'step_times', 1)
setattr(train_config, 'name', func_name)
setattr(train_config, 'ndim', out_ndim)
setattr(train_config, 'lrstep', 100) #learning rate decay epchoes
setattr(train_config, 'writer', writer)
setattr(train_config, 'func_main', func_main)

modelTrainer(train_config)
    
