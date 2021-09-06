# add modules folder to Python's search path
import sys
from pathlib import Path
from os.path import dirname, abspath
script_dir = Path(dirname(abspath('')))
module_dir = str(script_dir)
print(module_dir)
sys.path.insert(0, module_dir + '/modules')
sys.path.insert(0, module_dir + '/models')

import geomap as gm 
import numpy as np 
import Lorenz96_alt as lorenz

# initialize the generator
dim = 40
model, gen_path = lorenz.get_model(x0=np.random.uniform(size=dim), size=10, obs_gap=0.1)
data_folder = '../data'
data_gen = gm.DataGen(dim=dim, gen_path=gen_path, folder=data_folder)

# generate data
burn_in = 1000
num_paths = 50
length = 100
name = 'Lorenz96_{}_{}_{}'.format(dim, num_paths, length)
data_gen.create_random_dataset(num_paths=num_paths, length=length, burn_in=burn_in, name=name)
