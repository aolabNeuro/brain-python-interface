import unittest
import tasks
from analysis import performance
from tasks import generatorfunctions as genfns
import numpy as np 

from tasks import bmi_recon_tasks
from tasks.bmimultitasks import SimBMIControlMulti
reload(bmi_recon_tasks)
reload(tasks)

from tasks.bmi_recon_tasks import KFRMLCGRecon

cls = KFRMLCGRecon

idx = 5275

te = performance._get_te(idx, dbname='testing')
n_iter = len(te.hdf.root.task)
        
gen = SimBMIControlMulti.sim_target_seq_generator_multi(8, 1000)

task = cls(te, n_iter, gen)
from tasks import plantlist
task.plant = plantlist.cursor_14x14_no_vel_wall
task.init()
        
error = task.calc_recon_error(verbose=False)
abs_max_error = np.max(np.abs(error))

print abs_max_error