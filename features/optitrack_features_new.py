#=============================================================================
# Copyright © 2025 NaturalPoint, Inc. All Rights Reserved.
# 
# THIS SOFTWARE IS GOVERNED BY THE OPTITRACK PLUGINS EULA AVAILABLE AT https://www.optitrack.com/about/legal/eula.html 
# AND/OR FOR DOWNLOAD WITH THE APPLICABLE SOFTWARE FILE(S) (“PLUGINS EULA”). BY DOWNLOADING, INSTALLING, ACTIVATING 
# AND/OR OTHERWISE USING THE SOFTWARE, YOU ARE AGREEING THAT YOU HAVE READ, AND THAT YOU AGREE TO COMPLY WITH AND ARE
# BOUND BY, THE PLUGINS EULA AND ALL APPLICABLE LAWS AND REGULATIONS. IF YOU DO NOT AGREE TO BE BOUND BY THE PLUGINS
# EULA, THEN YOU MAY NOT DOWNLOAD, INSTALL, ACTIVATE OR OTHERWISE USE THE SOFTWARE AND YOU MUST PROMPTLY DELETE OR
# RETURN IT. IF YOU ARE DOWNLOADING, INSTALLING, ACTIVATING AND/OR OTHERWISE USING THE SOFTWARE ON BEHALF OF AN ENTITY,
# THEN BY DOING SO YOU REPRESENT AND WARRANT THAT YOU HAVE THE APPROPRIATE AUTHORITY TO ACCEPT THE PLUGINS EULA ON
# BEHALF OF SUCH ENTITY. See license file in root directory for additional governing terms and information.
#=============================================================================


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import sys
import time
import os
import numpy as np
from riglib.experiment import traits
from riglib import source
from features.neural_sys_features import CorticalBMI,CorticalData
import traceback

from riglib.optitrack_client_update.NatNetClient import NatNetClient, DataDescriptions, MoCapData
#import DataDescriptions
#import MoCapData

# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.
class OptiTrackBMI(CorticalBMI):
    """
    BMI using OptiTrack as the datasource.
    """ 
    optitrack_sampling_rate = traits.Float(120.0, desc="Sampling rate of the OptiTrack data")
    n_rigid_bodies = traits.Int(0, desc="Number of rigid bodies connected to the OptiTrack system")
    