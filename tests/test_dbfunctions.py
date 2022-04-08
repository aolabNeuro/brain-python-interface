import unittest
from db import dbfunctions as db
from datetime import datetime

class TestDBfcns(unittest.TestCase):

    def test_get_files_for_taskentry(self):
        this_date = datetime(2022, 3, 30)
        entries = db.get_task_entries(date=this_date)
        h, e = db.get_files_server(entries[0].get_data_files())
        self.assertEqual(h, '/mnt/hdf/beig20220330_01_te4604.hdf')
        self.assertEqual(e, '/mnt/ecube/2022-03-30_BMI3D_te4604')

if __name__ == '__main__':
    unittest.main()