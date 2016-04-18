'''
Classes here which inherit from django.db.models.Model define the structure of the database

Django database modules. See https://docs.djangoproject.com/en/dev/intro/tutorial01/
for a basic introduction
'''

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'db.settings'
import json
import cPickle, pickle
import inspect
from collections import OrderedDict
from django.db import models
from django.core.exceptions import ObjectDoesNotExist

import numpy as np

from riglib import calibrations, experiment
from config import config
import importlib
import subprocess    
import traceback
import imp

def _get_trait_default(trait):
    '''
    Function which tries to determine the default value for a trait in the class declaration
    '''
    _, default = trait.default_value()
    if isinstance(default, tuple) and len(default) > 0:
        try:
            func, args, _ = default
            default = func(*args)
        except:
            pass
    return default

class Task(models.Model):
    name = models.CharField(max_length=128)
    visible = models.BooleanField(default=True, blank=True)
    def __unicode__(self):
        return self.name
    
    def get(self, feats=()):
        print "models.Task.get()"
        from namelist import tasks

        if len(tasks) == 0: 
            print 'Import error in tracker.models.Task.get: from namelist import task returning empty -- likely error in task'
        
        feature_classes = Feature.getall(feats)

        if self.name in tasks and not None in feature_classes:
            try:
                # reload the module which contains the base task class
                task_cls = tasks[self.name]

                module_name = task_cls.__module__
                if '.' in module_name:
                    module_names = module_name.split('.')
                    mod = __import__(module_names[0])
                    for submod in module_names[1:]:
                        mod = getattr(mod, submod)
                else:
                    mod = __import__(mod_name)
                
                task_cls_module = mod
                task_cls_module = imp.reload(task_cls_module)
                task_cls = getattr(task_cls_module, task_cls.__name__)

                # run the metaclass constructor
                Exp = experiment.make(task_cls, feature_classes)
                return Exp
            except:
                print "Problem making the task class!"
                traceback.print_exc()
                print self.name
                print feats
                print Feature.getall(feats)
                print "*******"
                return experiment.Experiment
        elif self.name in tasks:
            return tasks[self.name]
        else:
            return experiment.Experiment

    @staticmethod
    def populate():
        '''
        Automatically create a new database record for any tasks added to db/namelist.py
        '''
        from namelist import tasks
        real = set(tasks.keys())
        db = set(task.name for task in Task.objects.all())
        for name in real - db:
            Task(name=name).save()

    def params(self, feats=(), values=None):
        '''

        Parameters
        ----------
        feats : iterable of Feature instances
            Features selected on the task interface
        values : dict
            Values for the task parameters

        '''
        #from namelist import instance_to_model, instance_to_model_filter_kwargs

        if values is None:
            values = dict()
        
        # Use an ordered dict so that params actually stay in the order they're added, instead of random (hash) order
        params = OrderedDict()

        # Run the meta-class constructor to make the Task class (base task class + features )
        Exp = self.get(feats=feats)
        ctraits = Exp.class_traits()

        def add_trait(trait_name):
            trait_params = dict()
            trait_params['type'] = ctraits[trait_name].trait_type.__class__.__name__
            trait_params['default'] = _get_trait_default(ctraits[trait_name])
            trait_params['desc'] = ctraits[trait_name].desc
            trait_params['hidden'] = 'hidden' if Exp.is_hidden(trait_name) else 'visible'

            if trait_name in values:
                trait_params['value'] = values[trait_name]

            if trait_params['type'] == "InstanceFromDB":
                # look up the model name in the trait
                mdl_name = ctraits[trait_name].bmi3d_db_model

                # get the database Model class from 'db.tracker.models'
                Model = globals()[mdl_name]
                filter_kwargs = ctraits[trait_name].bmi3d_query_kwargs

                # look up database records which match the model type & filter parameters
                insts = Model.objects.filter(**filter_kwargs).order_by("-date")
                trait_params['options'] = [(i.pk, i.path) for i in insts]

            elif trait_params['type'] == 'Instance':
                raise ValueError("You should use the 'InstanceFromDB' trait instead of the 'Instance' trait!")

            # if the trait is an enumeration, look in the 'Exp' class for 
            # the options because for some reason the trait itself can't 
            # store the available options (at least at the time this was written..)
            elif trait_params['type'] == "Enum":
                raise ValueError("You should use the 'OptionsList' trait instead of the 'Enum' trait!")

            elif trait_params['type'] == "OptionsList":
                trait_params['options'] = ctraits[trait_name].bmi3d_input_options

            elif trait_params['type'] == "DataFile":
                # look up database records which match the model type & filter parameters
                filter_kwargs = ctraits[trait_name].bmi3d_query_kwargs
                insts = DataFile.objects.filter(**filter_kwargs).order_by("-date")
                trait_params['options'] = [(i.pk, i.path) for i in insts]                

            params[trait_name] = trait_params

            if trait_name == 'bmi': # a hack for really old data, where the 'decoder' was mistakenly labeled 'bmi'
                params['decoder'] = trait_params

        # add all the traits that are explicitly instructed to appear at the top of the menu
        ordered_traits = Exp.ordered_traits
        for trait in ordered_traits:
            if trait in Exp.class_editable_traits():
                add_trait(trait)

        # add all the remaining non-hidden traits
        for trait in Exp.class_editable_traits():
            if trait not in params and not Exp.is_hidden(trait):
                add_trait(trait)

        # add any hidden traits
        for trait in Exp.class_editable_traits():
            if trait not in params:
                add_trait(trait)
        return params

    def sequences(self):
        from json_param import Parameters
        seqs = dict()
        for s in Sequence.objects.filter(task=self.id):
            seqs[s.id] = s.to_json()

        return seqs

    def get_generators(self):
        # Supply sequence generators which are declared to be compatible with the selected task class
        exp_generators = dict() 
        Exp = self.get()
        if hasattr(Exp, 'sequence_generators'):
            for seqgen_name in Exp.sequence_generators:
                try:
                    g = Generator.objects.using(self._state.db).get(name=seqgen_name)
                    exp_generators[g.id] = seqgen_name
                except:
                    print "missing generator %s" % seqgen_name
        return exp_generators        

class Feature(models.Model):
    name = models.CharField(max_length=128)
    visible = models.BooleanField(blank=True, default=True)
    def __unicode__(self):
        return self.name

    @property
    def desc(self):
        feature_cls = self.get()
        if not feature_cls is None:
            return feature_cls.__doc__
        else:
            return ''
    
    def get(self):
        from namelist import features
        if self.name in features:
            return features[self.name]
        else:
            return None

    @staticmethod
    def populate():
        from namelist import features
        real = set(features.keys())
        db = set(feat.name for feat in Feature.objects.all())
        for name in real - db:
            Feature(name=name).save()

    @staticmethod
    def getall(feats):
        features = []
        for feat in feats:
            if isinstance(feat, (int, float, str, unicode)):
                try:
                    feat = Feature.objects.get(pk=int(feat)).get()
                except ValueError:
                    try:
                        feat = Feature.objects.get(name=feat).get()
                    except:
                        print "Cannot find feature %s"%feat
                        continue
            elif isinstance(feat, models.Model):
                feat = feat.get()
            
            features.append(feat)
        return features

class System(models.Model):
    name = models.CharField(max_length=128)
    path = models.TextField()
    archive = models.TextField()

    def __unicode__(self):
        return self.name
    
    @staticmethod
    def populate():
        for name in ["eyetracker", "hdf", "plexon", "bmi", "bmi_params", "juice_log", "blackrock"]:
            try:
                System.objects.get(name=name)
            except ObjectDoesNotExist:
                System(name=name, path="/storage/rawdata/%s"%name).save()

    @staticmethod 
    def make_new_sys(name):
        try:
            new_sys_rec = System.objects.get(name=name)
        except ObjectDoesNotExist:
            data_dir = "/storage/rawdata/%s" % name
            new_sys_rec = System(name=name, path=data_dir)
            new_sys_rec.save()
            os.popen('mkdir -p %s' % data_dir)

        return new_sys_rec

    def save_to_file(self, obj, filename, obj_name=None, entry_id=-1):
        full_filename = os.path.join(self.path, filename)
        pickle.dump(obj, open(full_filename, 'w'))

        if obj_name is None:
            obj_name = filename.rstrip('.pkl')

        df = DataFile()
        df.path = filename 
        df.system = self 
        df.entry_id = entry_id
        df.save()


class Subject(models.Model):
    name = models.CharField(max_length=128)
    def __unicode__(self):
        return self.name

class Generator(models.Model):
    name = models.CharField(max_length=128)
    params = models.TextField()
    static = models.BooleanField()
    visible = models.BooleanField(blank=True, default=True)

    def __unicode__(self):
        return self.name
    
    def get(self):
        '''
        Retrieve the function that can be used to construct the ..... generator? sequence?
        '''
        from namelist import generators
        return generators[self.name]

    @staticmethod
    def populate():
        from namelist import generators
        listed_generators = set(generators.keys())
        db_generators = set(gen.name for gen in Generator.objects.all())

        # determine which generators are missing from the database using set subtraction
        missing_generators = listed_generators - db_generators
        for name in missing_generators:
            # The sequence/generator constructor can either be a callable or a class constructor... not aware of any uses of the class constructor
            try:
                args = inspect.getargspec(generators[name]).args
                print args
            except TypeError:
                args = inspect.getargspec(generators[name].__init__).args
                args.remove("self")
            
            # A generator is determined to be static only if it takes an "exp" argument representing the Experiment class
            static = ~("exp" in args)
            if "exp" in args:
                args.remove("exp")

            # TODO not sure why the 'length' argument is being removed; is it assumed that all generators will take a 'length' argument?
            if "length" in args:
                args.remove("length")

            gen_obj = Generator(name=name, params=",".join(args), static=static)
            gen_obj.save()

    def to_json(self, values=None):
        if values is None:
            values = dict()
        gen = self.get()
        try:
            args = inspect.getargspec(gen)
            names, defaults = args.args, args.defaults
        except TypeError:
            args = inspect.getargspec(gen.__init__)
            names, defaults = args.args, args.defaults
            names.remove("self")

        # if self.static:
        #     defaults = (None,)+defaults
        # else:
        #     #first argument is the experiment
        #     names.remove("exp")
        # arginfo = zip(names, defaults)

        params = OrderedDict()
        from itertools import izip
        for name, default in izip(names, defaults):
            if name == 'exp':
                continue
            typename = "String"

            params[name] = dict(type=typename, default=default, desc='')
            if name in values:
                params[name]['value'] = values[name]

        return dict(name=self.name, params=params)

class Sequence(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    generator = models.ForeignKey(Generator)
    name = models.CharField(max_length=128)
    params = models.TextField() #json data
    sequence = models.TextField(blank=True) #pickle data
    task = models.ForeignKey(Task)

    def __unicode__(self):
        return self.name
    
    def get(self):
        from riglib.experiment import generate
        from json_param import Parameters

        if hasattr(self, 'generator') and self.generator.static: # If the generator is static, (NOTE: the generator being static is different from the *sequence* being static)
            if len(self.sequence) > 0:
                return generate.runseq, dict(seq=cPickle.loads(str(self.sequence)))
            else:
                return generate.runseq, dict(seq=self.generator.get()(**Parameters(self.params).params))            
        else:
            return self.generator.get(), Parameters(self.params).params

    def to_json(self):
        from json_param import Parameters
        state = 'saved' if self.pk is not None else "new"
        js = dict(name=self.name, state=state)
        js['static'] = len(self.sequence) > 0
        js['params'] = self.generator.to_json(Parameters(self.params).params)['params']
        js['generator'] = self.generator.id, self.generator.name
        return js

    @classmethod
    def from_json(cls, js):
        '''
        Construct a models.Sequence instance from JSON data (e.g., generated by the web interface for starting experiments)
        '''
        from json_param import Parameters

        # Error handling when input argument 'js' actually specifies the primary key of a Sequence object already in the database
        try:
            seq = Sequence.objects.get(pk=int(js))
            print "retreiving sequence from POSTed ID"
            return seq
        except:
            pass
        
        # Make sure 'js' is a python dictionary
        if not isinstance(js, dict):
            js = json.loads(js)

        # Determine the ID of the "generator" used to make this sequence
        genid = js['generator']
        if isinstance(genid, (tuple, list)):
            genid = genid[0]
        
        # Construct the database record for the new Sequence object
        seq = cls(generator_id=int(genid), name=js['name'])

        # Link the generator instantiation parameters to the sequence record
        # Parameters are stored in JSON format in the database
        seq.params = Parameters.from_html(js['params']).to_json()

        # If the sequence is to be static, 
        if js['static']:
            print "db.tracker.models.Sequence.from_json: storing static sequence data to database"
            generator_params = Parameters(seq.params).params
            seq_data = seq.generator.get()(**generator_params)
            seq.sequence = cPickle.dumps(seq_data)
        return seq

class TaskEntry(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    task = models.ForeignKey(Task)
    feats = models.ManyToManyField(Feature)
    sequence = models.ForeignKey(Sequence, blank=True)

    params = models.TextField()
    report = models.TextField()
    notes = models.TextField()
    visible = models.BooleanField(blank=True, default=True)
    backup = models.BooleanField(blank=True, default=False)

    def __unicode__(self):
        return "{date}: {subj} on {task} task, id={id}".format(
            date=self.date.strftime("%h. %e, %Y, %l:%M %p"),
            subj=self.subject.name,
            task=self.task.name,
            id=self.id)
    
    def get(self, feats=()):
        from json_param import Parameters
        from riglib import experiment
        Exp = experiment.make(self.task.get(), tuple(f.get() for f in self.feats.all())+feats)
        params = Parameters(self.params)
        params.trait_norm(Exp.class_traits())
        if issubclass(Exp, experiment.Sequence):
            gen, gp = self.sequence.get()
            seq = gen(Exp, **gp)
            exp = Exp(seq, **params.params)
        else:
            exp = Exp(**params.params)
        exp.event_log = json.loads(self.report)
        return exp
    
    @property
    def task_params(self):
        from json_param import Parameters
        data = Parameters(self.params).params
        if 'bmi' in data:
            data['decoder'] = data['bmi']
        ##    del data['bmi']
        return data

    def plexfile(self, path='/storage/plexon/', search=False):
        rplex = Feature.objects.get(name='relay_plexon')
        rplexb = Feature.objects.get(name='relay_plexbyte')
        feats = self.feats.all()
        if rplex not in feats and rplexb not in feats:
            return None

        if not search:
            system = System.objects.get(name='plexon')
            df = DataFile.objects.filter(entry=self.id, system=system)
            if len(df) > 0:
                return df[0].get_path()
        
        if len(self.report) > 0:
            event_log = json.loads(self.report)
            import os, sys, glob, time
            if len(event_log) < 1:
                return None

            start = event_log[-1][2]
            files = sorted(glob.glob(path+"/*.plx"), key=lambda f: abs(os.stat(f).st_mtime - start))

            if len(files) > 0:
                tdiff = os.stat(files[0]).st_mtime - start
                if abs(tdiff) < 60:
                     return files[0]

    def offline_report(self):
        Exp = self.task.get(self.feats.all())
        
        if len(self.report) == 0:
            return dict()
        else:
            report = json.loads(self.report)
            rpt = Exp.offline_report(report)

            ## If this is a BMI block, add the decoder name to the report (doesn't show up properly in drop-down menu for old blocks)
            try:
                from db import dbfunctions
                te = dbfunctions.TaskEntry(self.id, dbname=self._state.db)
                rpt['Decoder name'] = te.decoder_record.name + ' (trained in block %d)' % te.decoder_record.entry_id
            except AttributeError:
                pass
            except:
                import traceback
                traceback.print_exc()
            return rpt

    def to_json(self):
        '''
        Create a JSON dictionary of the metadata associated with this block for display in the web interface
        '''
        print "starting TaskEntry.to_json()"
        from json_param import Parameters

        # Run the metaclass constructor for the experiment used. If this can be avoided, it would help to break some of the cross-package software dependencies,
        # making it easier to analyze data without installing software for the entire rig

        Exp = self.task.get(self.feats.all())        
        state = 'completed' if self.pk is not None else "new"

        js = dict(task=self.task.id, state=state, subject=self.subject.id, notes=self.notes)
        js['feats'] = dict([(f.id, f.name) for f in self.feats.all()])
        js['params'] = self.task.params(self.feats.all(), values=self.task_params)

        if len(js['params'])!=len(self.task_params):
            print 'param lengths: JS:', len(js['params']), 'Task: ', len(self.task_params)

        # Supply sequence generators which are declared to be compatible with the selected task class
        exp_generators = dict() 
        if hasattr(Exp, 'sequence_generators'):
            for seqgen_name in Exp.sequence_generators:
                try:
                    g = Generator.objects.using(self._state.db).get(name=seqgen_name)
                    exp_generators[g.id] = seqgen_name
                except:
                    print "missing generator %s" % seqgen_name
        js['generators'] = exp_generators

        ## Add the sequence, used when the block gets copied
        print "getting the sequence, if any"
        if issubclass(self.task.get(), experiment.Sequence):
            js['sequence'] = {self.sequence.id:self.sequence.to_json()}

        datafiles = DataFile.objects.using(self._state.db).filter(entry=self.id)

        ## Add data files linked to this task entry to the web interface. 
        try:
            backup_root = config.backup_root['root']
        except:
            backup_root = '/None'
        
        js['datafiles'] = dict()
        system_names = set(d.system.name for d in datafiles)
        for name in system_names:
            js['datafiles'][name] = [d.get_path() + ' (backup available: %s)' % d.is_backed_up(backup_root) for d in datafiles if d.system.name == name]

        js['datafiles']['sequence'] = issubclass(Exp, experiment.Sequence) and len(self.sequence.sequence) > 0
        
        # Parse the "report" data and put it into the JS response
        js['report'] = self.offline_report()

        if config.recording_sys['make'] == 'plexon':
            try:
                from plexon import plexfile # keep this import here so that only plexon rigs need the plexfile module installed
                plexon = System.objects.using(self._state.db).get(name='plexon')
                df = DataFile.objects.using(self._state.db).get(entry=self.id, system=plexon)

                _neuralinfo = dict(is_seed=Exp.is_bmi_seed)
                if Exp.is_bmi_seed:
                    plx = plexfile.openFile(str(df.get_path()), load=False)
                    path, name = os.path.split(df.get_path())
                    name, ext = os.path.splitext(name)

                    _neuralinfo['length'] = plx.length
                    _neuralinfo['units'] = plx.units
                    _neuralinfo['name'] = name

                js['bmi'] = dict(_neuralinfo=_neuralinfo)
            except MemoryError:
                print "Memory error opening plexon file!"
                js['bmi'] = dict(_neuralinfo=None)
            except (ObjectDoesNotExist, AssertionError, IOError):
                print "No plexon file found"
                js['bmi'] = dict(_neuralinfo=None)
        
        elif config.recording_sys['make'] == 'blackrock':
            try:
                length, units = parse_blackrock_file(self.nev_file, self.nsx_files)

                # Blackrock units start from 0 (unlike plexon), so add 1
                # for web interface purposes
                # i.e., unit 0 on channel 3 will be "3a" on web interface
                units = [(chan, unit+1) for chan, unit in units]

                js['bmi'] = dict(_neuralinfo=dict(
                    length=length, 
                    units=units,
                    name=name,
                    is_seed=int(Exp.is_bmi_seed),
                    ))    
            except (ObjectDoesNotExist, AssertionError, IOError):
                print "No blackrock files found"
                js['bmi'] = dict(_neuralinfo=None)
            except:
                import traceback
                traceback.print_exc()
                js['bmi'] = dict(_neuralinfo=None)
        elif config.recording_sys['make'] == 'TDT':
            raise NotImplementedError("This code does not yet know how to open TDT files!")
        else:
            raise Exception('Unrecognized recording_system!')


        for dec in Decoder.objects.using(self._state.db).filter(entry=self.id):
            js['bmi'][dec.name] = dec.to_json()

        # include paths to any plots associated with this task entry, if offline
        files = os.popen('find /storage/plots/ -name %s*.png' % self.id)
        plot_files = dict()
        for f in files:
            fname = f.rstrip()
            keyname = os.path.basename(fname).rstrip('.png')[len(str(self.id)):]
            plot_files[keyname] = os.path.join('/static', fname)

        js['plot_files'] = plot_files
        js['flagged_for_backup'] = self.backup
        js['visible'] = self.visible
        print "TaskEntry.to_json finished!"
        return js

    @property
    def plx_file(self):
        '''
        Returns the name of the plx file associated with the session.
        '''
        plexon = System.objects.get(name='plexon')
        try:
            df = DataFile.objects.get(system=plexon, entry=self.id)
            return os.path.join(df.system.path, df.path)
        except:
            import traceback
            traceback.print_exc()
            return 'noplxfile'

    @property
    def nev_file(self):
        '''
        Return the name of the nev file associated with the session.
        '''
        try:
            df = DataFile.objects.get(system__name="blackrock", path__endswith=".nev", entry=self.id)
            return df.get_path()
        except:
            import traceback
            traceback.print_exc()
            return 'no_nev_file'

    @property
    def nsx_files(self):
        '''Return a list containing the names of the nsx files (there could be more
        than one) associated with the session.
    
        nsx files extensions are .ns1, .ns2, ..., .ns6
        '''
        try:
            dfs = []
            for k in range(1, 7):
                df_k = DataFile.objects.filter(system__name="blackrock", path__endswith=".ns%d" % k, entry=self.id)
                dfs += list(df_k)

            return [df.get_path() for df in dfs]
        except:
            import traceback
            traceback.print_exc()
            return []

    @property
    def name(self):
        '''
        Return a string representing the 'name' of the block. Note that the block
        does not really have a unique name in the current implementation.
        Thus, the 'name' is a hack this needs to be hacked because the current way of determining a 
        a filename depends on the number of things in the database, i.e. if 
        after the fact a record is removed, the number might change. read from
        the file instead
        '''
        # import config
        if config.recording_sys['make'] == 'plexon':
            try:
                return str(os.path.basename(self.plx_file).rstrip('.plx'))
            except:
                return 'noname'
        elif config.recording_sys['make'] == 'blackrock':
            try:
                return str(os.path.basename(self.nev_file).rstrip('.nev'))
            except:
                return 'noname'
        else:
            raise Exception('Unrecognized recording_system!')

    @classmethod
    def from_json(cls, js):
        pass

    def get_decoder(self):
        """
        Get the Decoder instance associated with this task entry
        """
        params = eval(self.params)
        decoder_id = params['bmi']
        return Decoder.objects.get(id=decoder_id)

class Calibration(models.Model):
    subject = models.ForeignKey(Subject)
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    system = models.ForeignKey(System)

    params = models.TextField()

    def __unicode__(self):
        return "{date}:{system} calibration for {subj}".format(date=self.date, 
            subj=self.subject.name, system=self.system.name)
    
    def get(self):
        from json_param import Parameters
        return getattr(calibrations, self.name)(**Parameters(self.params).params)

class AutoAlignment(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.TextField()
    
    def __unicode__(self):
        return "{date}:{name}".format(date=self.date, name=self.name)
       
    def get(self):
        return calibrations.AutoAlign(self.name)


def decoder_unpickler(mod_name, kls_name):
    if kls_name == 'StateSpaceFourLinkTentacle2D':
        kls_name = 'StateSpaceNLinkPlanarChain'
        mod_name = 'riglib.bmi.state_space_models'

    if kls_name == 'StateSpaceEndptVel':
        kls_name = 'LinearVelocityStateSpace'
        mod_name = 'riglib.bmi.state_space_models'

    if kls_name == 'State':
        mod_name = 'riglib.bmi.state_space_models'
    mod = importlib.import_module(mod_name)
    return getattr(mod, kls_name)


class Decoder(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    name = models.CharField(max_length=128)
    entry = models.ForeignKey(TaskEntry)
    path = models.TextField()
    
    def __unicode__(self):
        return "{date}:{name} trained from {entry}".format(date=self.date, name=self.name, entry=self.entry)
    
    @property 
    def filename(self):
        data_path = getattr(config, 'db_config_%s' % self._state.db)['data_path']
        return os.path.join(data_path, 'decoders', self.path)        

    def load(self, db_name=None):
        if db_name is not None:
            data_path = getattr(config, 'db_config_'+db_name)['data_path']
        else:
            data_path = getattr(config, 'db_config_%s' % self._state.db)['data_path']
        decoder_fname = os.path.join(data_path, 'decoders', self.path)

        if os.path.exists(decoder_fname):
            fh = open(decoder_fname, 'r')
            unpickler = cPickle.Unpickler(fh)
            unpickler.find_global = decoder_unpickler
            dec = unpickler.load() # object will now contain the new class path reference
            fh.close()

            dec.name = self.name
            return dec
        else: # file not present!
            print "Decoder file could not be found! %s" % decoder_fname
            return None

    def get(self):
        return self.load()

    def to_json(self):
        dec = self.get()
        decoder_data = dict(name=self.name, path=self.path)
        if not (dec is None):
            decoder_data['cls'] = dec.__class__.__name__,
            if hasattr(dec, 'units'):
                decoder_data['units'] = dec.units
            else:
                decoder_data['units'] = []

            if hasattr(dec, 'binlen'):
                decoder_data['binlen'] = dec.binlen
            else:
                decoder_data['binlen'] = 0

            if hasattr(dec, 'tslice'):
                decoder_data['tslice'] = dec.tslice
            else:
                decoder_data['tslice'] = []

        return decoder_data

def parse_blackrock_file(nev_fname, nsx_files):
    '''
    # convert .nev file to hdf file using Blackrock's n2h5 utility (if it doesn't exist already)
    # this code goes through the spike_set for each channel in order to:
    #  1) determine the last timestamp in the file
    #  2) create a list of units that had spikes in this file
    '''
    nev_hdf_fname = nev_fname + '.hdf'

    if not os.path.isfile(nev_hdf_fname):
        subprocess.call(['n2h5', nev_fname, nev_hdf_fname])

    import tables #Previously import h5py -- pytables works fine too
    nev_hdf = tables.openFile(nev_hdf_fname, 'r')

    last_ts = 0
    units = []

    #for key in [key for key in nev_hdf.get('channel').keys() if 'channel' in key]:
    chans = nev_hdf.root.channel
    chan_names= chans._v_children
    for key in [key for key in chan_names.keys() if 'channel' in key]:
        chan_tab = nev_hdf.root.channel._f_getChild(key)
        if 'spike_set' in chan_tab:
            spike_set = chan_tab.spike_set
            if spike_set is not None:
                tstamps = spike_set[:]['TimeStamp']
                if len(tstamps) > 0:
                    last_ts = max(last_ts, tstamps[-1])
                else:
                    print 'skipping ', key, ': no spikes'

                channel = int(key[-5:])
                for unit_num in np.sort(np.unique(spike_set[:]['Unit'])):
                    units.append((channel, int(unit_num)))
        else:
            print 'skipping ', key, ': no spikeset'

    fs = 30000.
    nev_length = last_ts / fs
    nsx_lengths = []
    
    if nsx_files is not None:
        nsx_fs = dict()
        nsx_fs['.ns1'] = 500
        nsx_fs['.ns2'] = 1000
        nsx_fs['.ns3'] = 2000
        nsx_fs['.ns4'] = 10000
        nsx_fs['.ns5'] = 30000
        nsx_fs['.ns6'] = 30000

        NSP_channels = np.arange(128) + 1

        nsx_lengths = []
        for nsx_fname in nsx_files:

            nsx_hdf_fname = nsx_fname + '.hdf'
            if not os.path.isfile(nsx_hdf_fname):
                # convert .nsx file to hdf file using Blackrock's n2h5 utility
                subprocess.call(['n2h5', nsx_fname, nsx_hdf_fname])

            nsx_hdf = h5py.File(nsx_hdf_fname, 'r')

            for chan in NSP_channels:
                chan_str = str(chan).zfill(5)
                path = 'channel/channel%s/continuous_set' % chan_str
                if nsx_hdf.get(path) is not None:
                    last_ts = len(nsx_hdf.get(path).value)
                    fs = nsx_fs[nsx_fname[-4:]]
                    nsx_lengths.append(last_ts / fs)
                    
                    break

    length = max([nev_length] + nsx_lengths)
    return length, units, 


class DataFile(models.Model):
    date = models.DateTimeField(auto_now_add=True)
    local = models.BooleanField(default=True)
    archived = models.BooleanField(default=False)
    path = models.CharField(max_length=256)
    system = models.ForeignKey(System)
    entry = models.ForeignKey(TaskEntry)

    def __unicode__(self):
        if self.entry_id > 0:
            return "{name} datafile for {entry}".format(name=self.system.name, entry=self.entry)
        else:
            return "datafile '{name}' for System {sys_name}".format(name=self.path, sys_name=self.system.name)

    def to_json(self):
        return dict(system=self.system.name, path=self.path)

    def get(self):
        '''
        Open the datafile, if it's of a known type
        '''
        if self.system.name == 'hdf':
            import tables
            return tables.open_file(self.get_path())
        elif self.path[-4:] == '.pkl': # pickle file
            import pickle
            return pickle.load(open(self.get_path()))
        else:
            raise ValueError("models.DataFile does not know how to open this type of file: %s" % self.path)

    def get_path(self, check_archive=False):
        '''
        Get the full path to the file
        '''
        if not check_archive and not self.archived:
            text_file = open("path.txt", "w")
            text_file.write("path: %s" % os.path.join(self.system.path, self.path))
            text_file.close()
            return os.path.join(self.system.path, self.path)
        text_file2 = open("self.archive.txt", "w")
        text_file2.write("self.archive: %s" % self.archive)
        text_file2.close()
        
        paths = self.system.archive.split()
        text_file2 = open("paths.txt", "w")
        text_file2.write("paths: %s" % paths)
        text_file2.close()
        for path in paths:
            fname = os.path.join(path, self.path)
            if os.path.isfile(fname):
                text_file3 = open("fname.txt", "w")
                text_file3.write("fname: %s" % fname)
                text_file3.close()
                return fname

        raise IOError('File has been lost! '+fname)

    def has_cache(self):
        if self.system.name != "plexon":
            return False

        path, fname = os.path.split(self.get_path())
        fname, ext = os.path.splitext(fname)
        cache = os.path.join(path, '.%s.cache'%fname)
        return os.path.exists(cache)

    def remove(self, **kwargs):
        try:
            os.unlink(self.get_path())
        except OSError:
            print "already deleted..."

    def delete(self, **kwargs):
        self.remove()
        super(DataFile, self).delete(**kwargs)

    def is_backed_up(self, backup_root):
        '''
        Return a boolean indicating whether a copy of the file is available on the backup
        '''
        fname = self.get_path()
        rel_datafile = os.path.relpath(fname, '/storage')
        backup_fname = os.path.join(backup_root, rel_datafile)
        return os.path.exists(backup_fname)
