"""
MODIFIED to also correctly process 16-bit data (msgtype 2)

Simple Python RDA client for the RDA tcpip interface of the BrainVision Recorder
It reads all the information from the recorded EEG,
prints EEG and marker information to the console and calculates and
prints the average power every second


Brain Products GmbH
Gilching/Freiburg, Germany
www.brainproducts.com

"""

# BrainAmp MR has 16-bit A/D conversion
# see tech specs at: http://www.brainproducts.com/productdetails.php?id=5&tab=1
nbits = 16

if nbits == 16:    # should only receive data messages of type 2 (never 4)
    port = 51234
elif nbits == 32:  # should only receive data messages of type 4 (never 2)
    port = 51244
else:
    raise Exception('Invalid value for nbits -- must be either 16 or 32!')


# needs socket and struct library
from socket import *
from struct import *

# Marker class for storing marker information
class Marker:
    def __init__(self):
        self.position = 0
        self.points = 0
        self.channel = -1
        self.type = ""
        self.description = ""

# Helper function for receiving whole message
def RecvData(socket, requestedSize):
    returnStream = ''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            raise RuntimeError, "connection broken"
        returnStream += databytes
 
    return returnStream   

    
# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def SplitString(raw):
    stringlist = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != '\x00':
            s = s + raw[i]
        else:
            stringlist.append(s)
            s = ""

    return stringlist
    

# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = unpack('<d', rawdata[index:index+8])
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)

# # Helper function for extracting eeg and marker data from a raw data array
# # read from tcpip socket       
# def GetData(rawdata, channelCount):

#     # Extract numerical data
#     (block, points, markerCount) = unpack('<LLL', rawdata[:12])

#     # Extract eeg data as array of floats
#     data = []
#     for i in range(points * channelCount):
#         index = 12 + 4 * i
#         value = unpack('<f', rawdata[index:index+4])
#         data.append(value[0])

#     # Extract markers
#     markers = []
#     index = 12 + 4 * points * channelCount
#     for m in range(markerCount):
#         markersize = unpack('<L', rawdata[index:index+4])

#         ma = Marker()
#         (ma.position, ma.points, ma.channel) = unpack('<LLl', rawdata[index+4:index+16])
#         typedesc = SplitString(rawdata[index+16:index+markersize[0]])
#         ma.type = typedesc[0]
#         ma.description = typedesc[1]

#         markers.append(ma)
#         index = index + markersize[0]

#     return (block, points, markerCount, data, markers)


# MODIFIED VERSION of GetData that can be used for both 16-bit and 32-bit data
# Helper function for extracting eeg and marker data from a raw data array
# read from tcpip socket       
def GetData(rawdata, channelCount, nbits=16):

    # Extract numerical data
    (block, points, markerCount) = unpack('<LLL', rawdata[:12])

    if nbits == 16:
        fmt = '<h'  # little-endian byte order, signed 16-bit integer
        step = 2    # in bytes
    elif nbits == 32:
        fmt = '<f'  # little-endian byte order, 32-bit IEEE float
        step = 4    # in bytes
    else:
        raise Exception('Invalid value for nbits -- must be either 16 or 32!')

    # Extract eeg data as array of floats
    data = []
    for i in range(points * channelCount):
        # index = 12 + 4 * i  #  old
        # value = unpack('<f', rawdata[index:index+4])  # old
        index = 12 + (step * i)
        value = unpack(fmt, rawdata[index:index+step])
        data.append(value[0])

    # Extract markers
    markers = []
    # index = 12 + 4 * points * channelCount  # old
    index = 12 + (step * points * channelCount)
    for m in range(markerCount):
        markersize = unpack('<L', rawdata[index:index+4])

        ma = Marker()
        (ma.position, ma.points, ma.channel) = unpack('<LLl', rawdata[index+4:index+16])
        typedesc = SplitString(rawdata[index+16:index+markersize[0]])
        ma.type = typedesc[0]
        ma.description = typedesc[1]

        markers.append(ma)
        index = index + markersize[0]

    return (block, points, markerCount, data, markers)


##############################################################################################
#
# Main RDA routine
#
##############################################################################################

# Create a tcpip socket
con = socket(AF_INET, SOCK_STREAM)
# Connect to recorder host via 32Bit RDA-port
# adapt to your host, if recorder is not running on local machine
# change port to 51234 to connect to 16Bit RDA-port
# con.connect(("localhost", port))
con.connect(("192.168.137.1", port))

# Flag for main loop
finish = False

# data buffer for calculation, empty in beginning
data1s = []

# block counter to check overflows of tcpip buffer
lastBlock = -1

#### Main Loop ####
while not finish:

    # Get message header as raw array of chars
    rawhdr = RecvData(con, 24)

    # Split array into usefull information id1 to id4 are constants
    (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)
    print '(id1, id2, id3, id4, msgsize, msgtype)', (id1, id2, id3, id4, msgsize, msgtype)

    # Get data part of message, which is of variable size
    rawdata = RecvData(con, msgsize - 24)

    # Perform action dependend on the message type
    if msgtype == 1:
        # Start message, extract eeg properties and display them
        (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
        # reset block counter
        lastBlock = -1

        print "Start"
        print "Number of channels: " + str(channelCount)
        print "Sampling interval: " + str(samplingInterval)
        print "Resolutions: " + str(resolutions)
        print "Channel Names: " + str(channelNames)


    # NEW  -- original example client code did not process messages of type 2 (16-bit data)
    elif msgtype == 2:
        # Data message, extract data and markers
        # (block, points, markerCount, data, markers) = GetData(rawdata, channelCount)
        (block, points, markerCount, data, markers) = GetData(rawdata, channelCount, nbits=nbits)

        # Check for overflow
        if lastBlock != -1 and block > lastBlock + 1:
            print "*** Overflow with " + str(block - lastBlock) + " datablocks ***" 
        lastBlock = block

        # Print markers, if there are some in actual block
        if markerCount > 0:
            for m in range(markerCount):
                print "Marker " + markers[m].description + " of type " + markers[m].type

        # Put data at the end of actual buffer
        data1s.extend(data)

        # If more than 1s of data is collected, calculate average power, print it and reset data buffer
        if len(data1s) > channelCount * 1000000 / samplingInterval:
            index = int(len(data1s) - channelCount * 1000000 / samplingInterval)
            data1s = data1s[index:]

            for i in range(len(data1s)):
                print 'value:', data1s[i]*resolutions[i % channelCount]

            avg = 0
            # Do not forget to respect the resolution !!!
            for i in range(len(data1s)):
                avg = avg + data1s[i]*data1s[i]*resolutions[i % channelCount]*resolutions[i % channelCount]

            avg = avg / len(data1s)
            print "Average power: " + str(avg)

            data1s = []


    elif msgtype == 4:
        # Data message, extract data and markers
        # (block, points, markerCount, data, markers) = GetData(rawdata, channelCount)
        (block, points, markerCount, data, markers) = GetData(rawdata, channelCount, nbits=nbits)

        # Check for overflow
        if lastBlock != -1 and block > lastBlock + 1:
            print "*** Overflow with " + str(block - lastBlock) + " datablocks ***" 
        lastBlock = block

        # Print markers, if there are some in actual block
        if markerCount > 0:
            for m in range(markerCount):
                print "Marker " + markers[m].description + " of type " + markers[m].type

        # Put data at the end of actual buffer
        data1s.extend(data)

        # If more than 1s of data is collected, calculate average power, print it and reset data buffer
        if len(data1s) > channelCount * 1000000 / samplingInterval:
            index = int(len(data1s) - channelCount * 1000000 / samplingInterval)
            data1s = data1s[index:]

            for i in range(len(data1s)):
                print 'value:', data1s[i]*resolutions[i % channelCount]

            avg = 0
            # Do not forget to respect the resolution !!!
            for i in range(len(data1s)):
                avg = avg + data1s[i]*data1s[i]*resolutions[i % channelCount]*resolutions[i % channelCount]

            avg = avg / len(data1s)
            print "Average power: " + str(avg)

            data1s = []
            

    elif msgtype == 3:
        # Stop message, terminate program
        print "Stop"
        finish = True

# Close tcpip connection
con.close()
