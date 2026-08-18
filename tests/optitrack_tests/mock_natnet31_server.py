"""Mock NatNet 3.1 server (emulates Motive 2.x) for local verification.

Binds 127.0.0.1:1510, answers NAT_CONNECT with a ServerInfo advertising
NatNet protocol 3.1, answers NAT_REQUEST_MODELDEF with one marker set + one
rigid body description, and streams NatNet 3.1 frames at ~100 Hz to whoever
connected. Tracks StartRecording/StopRecording commands and reflects the
state in the frame suffix is_recording bit.

Usage (two terminals, no rig or Motive needed):
    python mock_natnet31_server.py
    python test_new_natnet_client.py --server 127.0.0.1 --unicast --duration 3 --record
"""
import socket
import struct
import sys
import time

NAT_CONNECT, NAT_SERVERINFO, NAT_REQUEST = 0, 1, 2
NAT_REQUEST_MODELDEF, NAT_MODELDEF, NAT_FRAMEOFDATA, NAT_KEEPALIVE = 4, 5, 7, 10


def packet(msg_id, payload):
    return struct.pack("<HH", msg_id, len(payload)) + payload


def server_info():
    return (b"MockMotive".ljust(256, b"\0")
            + bytes([2, 3, 0, 0])    # server (Motive) version 2.3
            + bytes([3, 1, 0, 0]))   # NatNet protocol 3.1


def model_def():
    p = struct.pack("<I", 2)  # dataset count (no per-dataset size field < 4.1)
    # type 0: marker set "Hand" with 2 named markers
    p += struct.pack("<I", 0) + b"Hand\0" + struct.pack("<I", 2) + b"m1\0m2\0"
    # type 1: rigid body "Hand", id 1, parent 0, offset, 2 markers (v3 layout:
    # positions block, then active-label block, no names)
    p += (struct.pack("<I", 1) + b"Hand\0" + struct.pack("<II", 1, 0)
          + struct.pack("<3f", 0.0, 0.0, 0.0)
          + struct.pack("<I", 2)
          + struct.pack("<3f", 0.01, 0.02, 0.03) + struct.pack("<3f", -0.01, -0.02, -0.03)
          + struct.pack("<II", 0, 0))
    return p


def frame(frame_number, recording):
    p = struct.pack("<I", frame_number)
    # marker sets: 1 set, "Hand", 2 markers
    p += struct.pack("<I", 1) + b"Hand\0" + struct.pack("<I", 2)
    p += struct.pack("<3f", 0.11, 0.22, 0.33) + struct.pack("<3f", 0.44, 0.55, 0.66)
    # unlabeled markers: 1
    p += struct.pack("<I", 1) + struct.pack("<3f", 1.0, 2.0, 3.0)
    # rigid bodies: 1 -- v3.x: id, pos, rot, mean marker error, params (bit0 = tracking valid)
    p += struct.pack("<I", 1)
    p += struct.pack("<I", 1) + struct.pack("<3f", 0.1, 0.2, 0.3)
    p += struct.pack("<4f", 0.0, 0.0, 0.0, 1.0) + struct.pack("<f", 0.0005)
    p += struct.pack("<H", 0x01)
    # skeletons: 0
    p += struct.pack("<I", 0)
    # labeled markers: 1 -- id, pos, size, params (2.6+), residual (3.0+)
    p += struct.pack("<I", 1)
    p += struct.pack("<I", (1 << 16) | 1) + struct.pack("<3f", 0.7, 0.8, 0.9)
    p += struct.pack("<f", 0.012) + struct.pack("<H", 0) + struct.pack("<f", 0.0002)
    # force plates: 0, devices: 0
    p += struct.pack("<II", 0, 0)
    # suffix: timecode, timecode_sub, timestamp (double, 2.7+), 3x uint64 hires (3.0+), params
    p += struct.pack("<II", 0, 0) + struct.pack("<d", frame_number / 100.0)
    p += struct.pack("<QQQ", 1000, 2000, 3000)
    p += struct.pack("<H", 0x01 if recording else 0x00)
    return p


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 1510))
    sock.setblocking(False)
    client_addr = None
    recording = False
    frame_number = 0
    deadline = time.time() + 30
    print("mock server listening on 127.0.0.1:1510", flush=True)
    while time.time() < deadline:
        # Drain everything pending (unicast clients send keepalives at a high rate),
        # capped so the frame sender below is never starved
        for _ in range(200):
            try:
                data, addr = sock.recvfrom(4096)
            except BlockingIOError:
                break
            msg_id = struct.unpack("<H", data[:2])[0]
            if msg_id == NAT_CONNECT:
                client_addr = addr
                sock.sendto(packet(NAT_SERVERINFO, server_info()), addr)
                print(f"client connected from {addr}", flush=True)
            elif msg_id == NAT_REQUEST_MODELDEF:
                sock.sendto(packet(NAT_MODELDEF, model_def()), addr)
            elif msg_id == NAT_REQUEST:
                cmd = data[4:].split(b"\0")[0].decode()
                print(f"command: {cmd}", flush=True)
                if cmd == "StartRecording":
                    recording = True
                elif cmd == "StopRecording":
                    recording = False
        if client_addr is not None:
            frame_number += 1
            sock.sendto(packet(NAT_FRAMEOFDATA, frame(frame_number, recording)), client_addr)
            time.sleep(0.01)
    print("mock server done", flush=True)


if __name__ == "__main__":
    main()
