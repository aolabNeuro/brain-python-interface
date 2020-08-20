import datetime
import pickle
import json
import traceback
import os

from django.template import RequestContext
from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
import urllib
import json
from django.http import HttpResponse
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import datetime as dt
import pdb

import io
import urllib, base64
from . import exp_tracker
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s

s.set(color_codes=True)
s.set_style("darkgrid")
s.set_context('paper')
s.set_palette("colorblind")

# Versions used
# print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
# print('Numpy: {}'.format(np.__version__))
# print('Matplotlib: {}'.format(m.__version__))
# print('Seaborn: {}'.format(s.__version__))

import h5py


def load(request):
    filename = '/Volumes/GoogleDrive/Shared drives/aoLab/Data/bmiLearning_jeev/jeev070412_072012/catBehaviorDat_jeev070412_072012.mat'
    with h5py.File(filename, "r") as f:
        # print("Keys: %s" %f.keys())
        data_import = {}

        for key_id in range(len(f.keys())):
            # a_group_key = list(f.keys())[key_id]
            var_suffix = list(f.keys())[key_id]
            # var_name = "data_"+ var_suffix
            data_import[var_suffix] = list(f[var_suffix])
    return render(request, 'analysis.html', dict=data_import)


def main(request):
    return render(request, "main.html", dict())


def plot_rastor(request):
    plt.plot(np.squeeze(data_epmTime), np.squeeze(data_epm[0, :]), 'b')
    plt.xlabel("epmTime")
    plt.ylabel("epm")
    plt.plot(np.squeeze(data_epmTime), np.squeeze(data_epm[1, :]), 'r')
    plt.plot(np.squeeze(data_epmTime), np.squeeze(data_epm[2, :]), 'g')
    plt.plot(np.squeeze(data_epmTime), np.squeeze(data_epm[3, :]), 'k')
    fig = plt.gcf()
    # plt.savefig(epmtimeVSepm,format= 'png')
    # string = base64.b64encode(buf.read)
    # uri=urllib.parse.quote(string)
    # plt.show()

    canvas = FigureCanvas(fig)
    response = HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response
