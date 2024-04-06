import time
from riglib import experiment
from analysis import online_analysis
from features.debug_features import OnlineAnalysis
import unittest
import numpy as np
import os
import threading

class TestOnlineAnalysis(unittest.TestCase):

    def test_basic_experiment(self):

        analysis = online_analysis.OnlineDataServer('localhost', 5000)

        Exp = experiment.make(experiment.Experiment, feats=[OnlineAnalysis])
        exp = Exp(fps=1, session_length=5, online_analysis_ip='localhost', online_analysis_port=5000)
        print(exp.dtype)
        exp.init()

        analysis.update()
        self.assertEqual(analysis.task_params['experiment_name'], 'Experiment')

        threading.Thread(target=exp.run).start()
        time.sleep(1)
        analysis.update()
        self.assertTrue(analysis.is_running)
        self.assertFalse(analysis.is_completed)
        self.assertEqual(analysis.state, 'wait')

        time.sleep(6)
        analysis.update()
        self.assertTrue(analysis.is_completed)
        self.assertFalse(analysis.is_running)
        self.assertEqual(analysis.state, 'None')
        self.assertEqual(analysis.cycle_count, 4)

        analysis.close()


if __name__ == '__main__':
    unittest.main()