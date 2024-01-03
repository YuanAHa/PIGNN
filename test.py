import torch
from core.utils.tools import parse_config, modelTester, RemoveDir
from core.utils.tools import rollout_error_test, plot_error_curve, render_results
from core.models import msgPassing
from core.geometry import RectangleMesh
import os


# func_name = 'heat'
# heat_params = [4. , 6. , 0.5 , 0.3]
# out_ndim = 1

func_name = 'burgers'
burgers_R = 60
out_ndim = 2

# func_name = 'fn'
# out_ndim = 2


delta_t = 1e-3
ckptpath = 'checkpoint/simulator_%s.pth' % func_name
device = torch.device(0)

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

bc = func_main.boundary_condition
ic = func_main.init_condition

dens = 100
area = '[00,11]'


def get_test_config(graph, **kwargs):     
       
    test_config = parse_config()
    model = kwargs['model']
    setattr(test_config, 'Reynold', burgers_R)
    # setattr(test_config, 'params', heat_params)   
    setattr(test_config, 'delta_t', delta_t)
    setattr(test_config, 'device', device)   
    setattr(test_config, 'ic', ic)
    setattr(test_config, 'bc', bc)
    setattr(test_config, 'model', model)
    setattr(test_config, 'test_steps', test_steps)
    setattr(test_config, 'NodeTypesRef', RectangleMesh.node_type_ref)
    setattr(test_config, 'name', func_name)
    setattr(test_config, 'ndim', out_ndim)
    setattr(test_config, 'graph_modify', func_main.graph_modify)
    setattr(test_config, 'graph', graph)
    setattr(test_config, 'density', dens)
    setattr(test_config, 'area', area)

    return test_config    

#-----------------------------------------
mesh = RectangleMesh(ru=(1, 1), lb=(0, 0), density=10)
graph = mesh.getGraphData()
model = msgPassing(message_passing_num=1, node_input_size=3+out_ndim, 
                   edge_input_size=3, ndim=out_ndim, device=device, model_dir=ckptpath)
model.load_model(ckptpath)
model.to(device)
model.eval()
test_steps = 10

test_config  = get_test_config(graph.to(device), model=model)

print('************* model test starts! ***********************')
predict_results = modelTester(test_config)

real_results = []
for step in range(1, test_config.test_steps +1):
    t = step * delta_t
    v1 = func_main.exact_solution(graph.pos, t)
    real_results.append(v1)
real_results = torch.stack(real_results, dim=0).cpu().numpy()


aRMSE = rollout_error_test(predict_results, real_results) 


#-----------------plotting----------------------------

results_root = 'PIGNN_%s_Results/'%(test_config.name)

aRMSE_Fig_save_dir = results_root + 'aRMSE_Fig/'
os.makedirs(aRMSE_Fig_save_dir, exist_ok = True)
print('PIGNN_%s_Reynold[%d]_area%s_dens[%d]_Steps[%d]: [loss_mean: %.4e]'%(
    test_config.name, test_config.Reynold, test_config.area, test_config.density, test_config.test_steps, aRMSE[-1]))
plot_error_curve(aRMSE, 0, test_config, aRMSE_Fig_save_dir)



testImg_save_dir = results_root + 'testImages_%s_Reynold[%d]_area%s_dens[%d]_Steps[%d]_ALL/'%(\
    test_config.name, test_config.Reynold, test_config.area, test_config.density, test_config.test_steps)
RemoveDir(testImg_save_dir)


render_results(predict_results, real_results, test_config, testImg_save_dir)