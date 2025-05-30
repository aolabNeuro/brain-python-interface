import sys

sys.path.append('/home/aolab/NatNet_SDK_4.2_ubuntu/samples/PythonClient')
import time

from NatNetClient import NatNetClient
from riglib.optitrack_client_update_natnet.optitrack_client_updated import *
from features.optitrack_updated import * 
import DataDescriptions
import MoCapData


def set_take(client, take_name):
        sz_command="SetRecordTakeName," + take_name
        return_code = client.send_command(sz_command)
        time.sleep(1)
        print("Command: %s - return_code: %d"% (sz_command, return_code) )

        
        #return self._send_command_and_wait("SetRecordTakeName," + take_name)

def receive_rigid_body_frame(new_id, position, rotation):
    print(f"callback triggered for rigid body {new_id}")
    if position:
        x,y,z = position
        print(f"Rigid Body ID {new_id}: Position -> x: {x:.3f}, y:{y:.3f}, z: {z:.3f}")

def receive_new_frame(data_dict):
    order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
    dump_args = False
    if dump_args == True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict :
                out_string += data_dict[key] + " "
            out_string+="/"
        print(out_string)

def request_data_descriptions(s_client):
    # Request the model definitions
    s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF,    "",  (s_client.server_ip_address, s_client.command_port) )

streaming_client = NatNetClient()
streaming_client.rigid_body_listener = receive_rigid_body_frame

if streaming_client.rigid_body_listener:
    print('registerd the listener')
else:
    print('failed to register the listener')

#streaming_client.new_frame_listener = receive_new_frame

# Start up the streaming client now that the callbacks are set up.
# This will run perpetually, and operate on a separate thread.
optionsDict = {}
optionsDict["stream_type"] = 'd'

#set_take(streaming_client, 'take_name')

is_running = streaming_client.run(optionsDict["stream_type"])
if not is_running:
    print("ERROR: Could not start streaming client.")
    try:
        sys.exit(1)
    except SystemExit:
        print("...")
    finally:
        print("exiting")

is_looping = True
time.sleep(1)
if streaming_client.connected() is False:
    print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
    try:
        sys.exit(2)
    except SystemExit:
        print("...")
    finally:
        print("exiting")
elif streaming_client.connected() is True:
    print("Successfully connected")

try:
    #request_data_descriptions(streaming_client)
    print("streaming rigid body data... Press ctrl_c to stop.")
    while True:
            time.sleep(1)
except KeyboardInterrupt:
     print("stopping....")
     streaming_client.shutdown()