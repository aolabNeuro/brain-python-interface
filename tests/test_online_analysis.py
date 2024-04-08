import time
from riglib import experiment
from analysis import online_analysis
from features.debug_features import OnlineAnalysis
import unittest
import numpy as np
import os
import threading

class TestOnlineAnalysis(unittest.TestCase):

    def test_manual_run(self):

        analysis = online_analysis.OnlineDataServer('localhost', 5000)

        Exp = experiment.make(experiment.Experiment, feats=[OnlineAnalysis])
        exp = Exp(fps=1, session_length=5, online_analysis_ip='localhost', online_analysis_port=5000)
        print(exp.dtype)
        exp.init()

        time.sleep(1)
        while True:
            if not analysis.update():
                break
        self.assertEqual(analysis.task_params['experiment_name'], 'Experiment')

        threading.Thread(target=exp.run).start()
        time.sleep(1)

        while True:
            if not analysis.update():
                break
        self.assertTrue(analysis.is_running)
        self.assertFalse(analysis.is_completed)
        self.assertEqual(analysis.state, 'wait')

        time.sleep(6)
        while True:
            if not analysis.update():
                break
        self.assertTrue(analysis.is_completed)
        self.assertFalse(analysis.is_running)
        self.assertEqual(analysis.state, None)

        analysis._stop()

    @unittest.skip("")
    def test_threaded(self):

        analysis = online_analysis.OnlineDataServer('localhost', 5000)

        # Start exp 1
        Exp = experiment.make(experiment.Experiment, feats=[OnlineAnalysis])
        exp = Exp(fps=1, session_length=5, online_analysis_ip='localhost', online_analysis_port=5000)
        print(exp.dtype)
        exp.init()

        # Start analysis
        analysis.start()
        time.sleep(1)
        threading.Thread(target=exp.run).start()

        time.sleep(6)

        # Start exp 2
        Exp = experiment.make(experiment.Experiment, feats=[OnlineAnalysis])
        exp = Exp(fps=1, session_length=5, online_analysis_ip='localhost', online_analysis_port=5000)
        print(exp.dtype)
        exp.init()
        time.sleep(1)
        threading.Thread(target=exp.run).start()
        time.sleep(6)

        # Wrap up
        analysis.stop()
        analysis.join()


if __name__ == '__main__':
    unittest.main()