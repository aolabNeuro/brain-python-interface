from db.tracker.models import TaskEntry, Feature, Sequence, Task, Subject, Experimenter
from riglib.experiment.scripting import run_experiment

def test():
    # Test the run_experiment function
    exp = Task.objects.all()[0]
    seq = Sequence.objects.all()[0]
    subject = Subject.objects.all()[0]
    experimenter = Experimenter.objects.all()[0]
    print('Experiment:', exp.name)
    print('Sequence:', seq.name)
    feat_names = ['optitrack']
    params = {
        'offset': [-20, -95, 0],
        'session_length': 30, 
    }
    run_experiment(subject.id, experimenter.id, 'project', 'session', exp.id, feat_names, seq.id, **params)


if __name__ == '__main__':
    test()