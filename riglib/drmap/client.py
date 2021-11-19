import anvil.server

# This is a sample client file demonstrating how to call Anvil annotated server-side methods.
# Method 1 uses anvil uplink https://anvil.works/docs/uplink
# Method 2 uses http (via anvil annotations)
APP_HOSTNAME = "localhost"
APP_PORT = "3030"

uplink_url = f"ws://{APP_HOSTNAME}:{APP_PORT}/_/uplink"

# Connects to anvil project on host url with given uplink key. Calls a registered method
# Requires anvil library & uplink key
def init():
    anvil.server.connect("FPTDBWQGPTHNSZFYBSH6K76P-OJFXNIBCJSFEZQRJ-CLIENT", url=uplink_url)
    ws_resp = anvil.server.call('test_websocket')
    print(ws_resp)

# Live test
def mtypes():
    ws_resp = anvil.server.call('get_mapping_types')
    print(ws_resp)

def get_most_recent_mappings(model, filters):
    # pretend for now, haven't quite figured out how to do this yet
    ws_resp = anvil.server.call('get_mapping_types')
    return ws_resp

def get_mapping(name):
    ws_resp = anvil.server.call('get_mapping', name)
    return ws_resp
