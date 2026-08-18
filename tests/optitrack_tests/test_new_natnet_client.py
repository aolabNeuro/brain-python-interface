#!/usr/bin/env python
"""
Standalone verification script for the TimSchneider42 python-natnet-client
(https://github.com/TimSchneider42/python-natnet-client, `pip install natnet==0.3.0`),
which is being evaluated as a replacement for the leoscholl/python_natnet client
currently referenced by features/optitrack_features.py.

This script does NOT import anything from bmi3d, so it can be copied to and run on
any machine that can reach the Motive PC. It exercises everything bmi3d needs:

    1. connect + server/protocol version detection (key evidence that old
       Motive 2.x / NatNet 3.x rigs still work)
    2. data descriptions (rigid body names/IDs)
    3. frame streaming (rate, rigid body positions, tracking_valid)
    4. (--record only) the Motive recording command workflow used by
       features/optitrack_features.py, verified via the frame is_recording bit

Known rig addresses (config/rig_defaults.py):
    pagaiisland2  10.155.206.1
    siberut-bmi   10.155.204.10
    human-bmi     128.95.215.191

Examples:
    python test_new_natnet_client.py --server 10.155.204.10
    python test_new_natnet_client.py --server 128.95.215.191 --unicast --duration 5
    python test_new_natnet_client.py --server 10.155.204.10 --record --take "natnet client test"

Expected result on a Motive 2.x rig: NatNet protocol version 3.x reported and all
steps pass. On a Motive 3+ rig: protocol version 4.x reported and all steps pass.
"""

import argparse
import socket
import sys
import time

try:
    from natnet import NatNetClient
except ImportError:
    sys.exit(
        "The 'natnet' package (TimSchneider42/python-natnet-client) is not installed.\n"
        "Install it with: pip install natnet==0.3.0\n"
        "Note: if the old leoscholl/python_natnet package is installed, uninstall it "
        "first -- both use the module name 'natnet'."
    )


def guess_local_ip(server_ip):
    """Find the local interface IP that routes to the server (UDP connect trick)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((server_ip, 1510))
        return s.getsockname()[0]
    finally:
        s.close()


class StepReporter:

    def __init__(self):
        self.failures = 0

    def report(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f": {detail}" if detail else ""))
        if not ok:
            self.failures += 1
        return ok


def pump_frames(client, duration, frames):
    """Pump the client synchronously for `duration` seconds, collecting frames."""
    n_start = len(frames)
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration:
        client.update_sync()
        # In unicast mode every update_sync() call sends a keepalive packet to
        # Motive, so don't poll too fast; pending frames are all drained per call.
        time.sleep(0.005)
    return len(frames) - n_start, time.perf_counter() - t_start


def wait_for_recording_state(client, frames, desired, timeout=5.0):
    """Pump until the latest frame's is_recording bit matches `desired`."""
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < timeout:
        client.update_sync()
        if frames and frames[-1].suffix.is_recording == desired:
            return True
        time.sleep(0.01)
    return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Known rig addresses"):],
    )
    parser.add_argument("--server", required=True, help="IP of the PC running Motive")
    parser.add_argument("--local-ip", default=None,
                        help="IP of this machine's interface facing the Motive PC "
                             "(default: auto-detected; required to be correct for multicast)")
    parser.add_argument("--unicast", action="store_true",
                        help="use unicast instead of multicast (Motive streaming pane "
                             "must be set to unicast too)")
    parser.add_argument("--multicast-address", default="239.255.42.99",
                        help="multicast group configured in Motive (default %(default)s)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds to stream frames for (default %(default)s)")
    parser.add_argument("--record", action="store_true",
                        help="also test the recording command workflow "
                             "(LiveMode / SetRecordTakeName / StartRecording / StopRecording). "
                             "This records a short throwaway take in Motive!")
    parser.add_argument("--take", default="natnet client test take",
                        help="take name to use with --record (default %(default)s)")
    args = parser.parse_args()

    local_ip = args.local_ip or guess_local_ip(args.server)
    print(f"Server: {args.server}   Local: {local_ip}   "
          f"Mode: {'unicast' if args.unicast else 'multicast ' + args.multicast_address}")

    reporter = StepReporter()
    frames = []
    descriptions = []

    client = NatNetClient(
        server_ip_address=args.server,
        local_ip_address=local_ip,
        multicast_address=args.multicast_address,
        use_multicast=not args.unicast,
    )
    client.on_data_frame_received_event.handlers.append(frames.append)
    client.on_data_description_received_event.handlers.append(descriptions.append)

    # Step 1: connect and report versions
    try:
        client.connect(timeout=5)
    except (TimeoutError, OSError) as e:
        reporter.report("connect", False,
                        f"{type(e).__name__}: {e} -- is Motive open and streaming enabled?")
        sys.exit(1)
    info = client.server_info
    reporter.report(
        "connect", True,
        f"{info.application_name} {info.server_version}, "
        f"NatNet protocol {info.nat_net_protocol_version}")

    try:
        # Step 2: data descriptions
        client.request_modeldef()
        t_start = time.perf_counter()
        while not descriptions and time.perf_counter() - t_start < 5:
            client.update_sync()
            time.sleep(0.01)
        if reporter.report("data descriptions", bool(descriptions)):
            desc = descriptions[-1]
            for rb in desc.rigid_bodies:
                print(f"       rigid body id={rb.id_num} name={rb.name!r}")
            if not desc.rigid_bodies:
                print("       (no rigid bodies defined in Motive -- streaming test "
                      "will have nothing to report)")

        # Step 3: frame streaming
        n_frames, elapsed = pump_frames(client, args.duration, frames)
        rate = n_frames / elapsed if elapsed > 0 else 0
        reporter.report("frame streaming", n_frames > 0,
                        f"{n_frames} frames in {elapsed:.1f} s ({rate:.0f} Hz)")
        if frames:
            stats = {}  # id -> [n_seen, n_valid, last_pos]
            for frame in frames:
                for rb in frame.rigid_bodies:
                    entry = stats.setdefault(rb.id_num, [0, 0, None])
                    entry[0] += 1
                    # tracking_valid is None on NatNet < 2.6
                    entry[1] += 1 if rb.tracking_valid else 0
                    entry[2] = rb.pos
            for rb_id, (n_seen, n_valid, pos) in sorted(stats.items()):
                pos_str = "(" + ", ".join(f"{x:.4f}" for x in pos) + ")"
                print(f"       rigid body id={rb_id}: {n_seen} frames, "
                      f"{100 * n_valid / n_seen:.0f}% tracking_valid, last pos {pos_str} m")
            if not stats:
                print("       (frames received but they contain no rigid bodies)")
            n_markers = len(frames[-1].labeled_markers or ())
            print(f"       last frame: {n_markers} labeled markers, "
                  f"timestamp {frames[-1].suffix.timestamp:.3f}")

        # Step 4: recording commands (optional; mirrors features/optitrack_features.py)
        if args.record:
            client.send_command("LiveMode")
            time.sleep(0.1)
            client.send_command(f"SetRecordTakeName,{args.take}")
            client.send_command("StartRecording")
            reporter.report("start recording",
                            wait_for_recording_state(client, frames, True),
                            f"is_recording bit set, take {args.take!r}")
            time.sleep(1)
            client.send_command("StopRecording")
            reporter.report("stop recording",
                            wait_for_recording_state(client, frames, False),
                            "is_recording bit cleared")
    finally:
        client.shutdown()

    if reporter.failures:
        print(f"\n{reporter.failures} step(s) FAILED")
        sys.exit(1)
    print("\nAll steps passed")


if __name__ == "__main__":
    main()
