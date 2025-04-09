import time
import traceback
import numpy as np
import os
import socket

from db.tracker.models import TaskEntry, Feature, Sequence, Task, Subject, Experimenter
from db.tracker.json_param import Parameters

from .. import experiment
from .task_wrapper import TaskWrapper

def run_experiment(subject_id, experimenter_id, project, session, 
                     task_id, feat_names, seq_id=None, task_desc=None, **kwargs):
    '''
    Run a task with the given database IDs and parameters. Just like running an experiment in the GUI,
    the task is added to the database. This is a blocking call.
    '''
    hostname = socket.gethostname()
    if hostname in ['pagaiisland2', 'human-bmi']:
        os.environ['DISPLAY'] = ':0.1'

    task =  Task.objects.get(pk=task_id)
    subject = Subject.objects.get(pk=subject_id)
    experimenter = Experimenter.objects.get(pk=experimenter_id)

    entry = TaskEntry.objects.create(rig_name=hostname, subject_id=subject.id, task_id=task.id, experimenter_id=experimenter.id,
        project=project, session=session)
    entry.entry_name = task_desc

    # Link the features used to the task entry
    for feat_name in feat_names:
        f = Feature.objects.get(name=feat_name)
        entry.feats.add(f.pk)

    js_params = entry.task.params(entry.feats.all(), values=kwargs)
    params = {k: js_params[k]['value'] if 'value' in js_params[k] else js_params[k]['default'] for k in js_params.keys()}
    params = Parameters.from_dict(params)
    print('params:', params.to_json())
    entry.params = params.to_json()

    kwargs = dict(subj=entry.subject.id, subject_name=subject.name)

    # Save the target sequence to the database and link to the task entry, if the task type uses target sequences
    if issubclass(task.get(feats=feat_names), experiment.Sequence):
        seq = Sequence.objects.get(pk=seq_id)
        entry.sequence = seq
        kwargs['seq'] = seq

    # Save the task entry to database
    entry.sw_version = 'script'
    entry.save()

    # Give the entry ID to the runtask as a kwarg so that files can be linked after the task is done
    kwargs['saveid'] = entry.id

    # Start the task FSM and tracker
    try:
        if 'seq' in kwargs:
            kwargs['seq_params'] = kwargs['seq'].params
            kwargs['seq'] = kwargs['seq'].get()  ## retreive the database data on this end of the pipe
        task_class = experiment.make(task.get(), feats=entry.feats.all())
        params.trait_norm(task_class.class_traits())
        params = params.params

        # For now we are running the task in the same process as the tracker
        # and not using the RPC as intended
        exp = TaskWrapper(params=params, target_class=task_class, websock=None, **kwargs)
        exp.target_constr()
        exp.target_destr(0, '')

    except:
        print("Error starting task:", kwargs)
        traceback.print_exc()