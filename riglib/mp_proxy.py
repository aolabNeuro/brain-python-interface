import time
import queue
import inspect
import traceback
import multiprocessing as mp
import io


class PipeWrapper(object):
    """Legacy base class kept for backward compatibility."""
    def __init__(self, pipe=None, log_filename='', cmd_event=None, **kwargs):
        self.pipe = pipe
        self.log_filename = log_filename
        self.cmd_event = cmd_event

    def log_error(self, err, mode='a'):
        if self.log_filename != '':
            traceback.print_exc(None, err)
            with open(self.log_filename, mode) as fp:
                err.seek(0)
                fp.write(err.read())

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)


class FuncProxy:
    '''
    Interface for calling functions in remote processes via multiprocessing Queues.
    '''
    def __init__(self, name, req_queue, resp_queue, lock, log_filename=''):
        '''
        Constructor for FuncProxy

        Parameters
        ----------
        name : string
            Name of remote function to call
        req_queue : mp.Queue
            Queue through which to send (function name, arguments)
        resp_queue : mp.Queue
            Queue from which to receive the result
        lock : mp.Lock
            Lock to serialize concurrent calls from the same process

        Returns
        -------
        FuncProxy instance
        '''
        self.name = name
        self.req_queue = req_queue
        self.resp_queue = resp_queue
        self.lock = lock
        self.log_filename = log_filename

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)

    def __call__(self, *args, **kwargs):
        '''
        Return the result of the remote function call.

        Parameters
        ----------
        *args, **kwargs
            Passed to the remote function

        Returns
        -------
        function result
        '''
        self.lock.acquire()
        self.log_str(f'lock acquired for {self.name}')
        try:
            self.req_queue.put((self.name, args, kwargs))
            try:
                resp = self.resp_queue.get(timeout=10)
            except queue.Empty:
                raise TimeoutError(
                    f"FuncProxy: remote process did not respond to '{self.name}'"
                )
            if isinstance(resp, Exception):
                raise resp
            return resp
        finally:
            self.lock.release()
            self.log_str(f'lock released for {self.name}')


class ObjProxy:
    '''
    Proxy for an object running in a remote process.
    Uses multiprocessing Queues for communication, which are safe
    for both fork and spawn process start methods.
    '''
    def __init__(self, target_class, req_queue, resp_queue, log_filename=''):
        object.__setattr__(self, '_target_class', target_class)
        object.__setattr__(self, '_req_queue', req_queue)
        object.__setattr__(self, '_resp_queue', resp_queue)
        object.__setattr__(self, '_log_filename', log_filename)
        # mp.Lock is picklable, needed when this proxy is passed to spawned processes
        object.__setattr__(self, '_lock', mp.Lock())

        is_instance_method = lambda n: inspect.isfunction(getattr(target_class, n))
        methods = set(filter(is_instance_method, dir(target_class)))
        object.__setattr__(self, 'methods', methods)

    def log_str(self, s, mode="a", newline=True):
        log_filename = object.__getattribute__(self, '_log_filename')
        if log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(log_filename, mode) as fp:
                fp.write(s)

    def _make_func_proxy(self, name):
        req_queue = object.__getattribute__(self, '_req_queue')
        resp_queue = object.__getattribute__(self, '_resp_queue')
        lock = object.__getattribute__(self, '_lock')
        log_filename = object.__getattribute__(self, '_log_filename')
        return FuncProxy(name, req_queue, resp_queue, lock, log_filename=log_filename)

    def __getattr__(self, attr):
        self.log_str(f"remotely getting attribute: {attr}")
        methods = object.__getattribute__(self, 'methods')
        if attr in methods:
            self.log_str(f"returning function proxy for {attr}")
            return self._make_func_proxy(attr)
        else:
            self.log_str("sending __getattribute__ over queue")
            return self._make_func_proxy('__getattribute__')(attr)

    def set(self, attr, value):
        self.log_str(f"ObjProxy.set: {attr} = {value}")
        self._make_func_proxy('__setattr__')(attr, value)
        self.log_str(f"Finished setting remote attr {attr} to {value}")

    def terminate(self):
        req_queue = object.__getattribute__(self, '_req_queue')
        req_queue.put(None)


class DataPipe(PipeWrapper):
    pass


def call_from_remote(x):
    return x

def call_from_parent(x):
    return x


class RPCProcess(mp.Process):
    """mp.Process which implements remote procedure call (RPC) via multiprocessing Queues.

    Uses mp.Queue for IPC instead of mp.Pipe + mp.Event, which avoids
    deadlocks when using the 'spawn' process start method (default on macOS).
    """
    def __init__(self, target_class=object, target_kwargs=dict(), log_filename='', **kwargs):
        super().__init__()
        self.log_filename = log_filename

        self.target = None
        self.target_class = target_class
        self.target_kwargs = target_kwargs

        self.status = mp.Value('b', 1)  # shared flag for terminating the remote process

        # Queues for RPC – picklable and spawn-safe
        self._req_queue = mp.Queue()
        self._resp_queue = mp.Queue()

        self.target_proxy = None
        self.data_proxy = None

    def __getattr__(self, attr):
        """Redirect attribute access to the target object when not found locally."""
        try:
            target_proxy = object.__getattribute__(self, 'target_proxy')
            status = object.__getattribute__(self, 'status')
            if target_proxy is not None and status.value > 0:
                try:
                    return getattr(target_proxy, attr)
                except Exception:
                    raise AttributeError(
                        f"RPCProcess: could not forward getattr '{attr}' to target"
                    )
            else:
                raise AttributeError("RPCProcess: target proxy not initialized")
        except AttributeError:
            raise
        except Exception:
            raise AttributeError(f"Could not get RPCProcess attribute: {attr}")

    def log_error(self, err=None, mode='a'):
        if self.log_filename != '':
            with open(self.log_filename, mode) as fp:
                traceback.print_exc(file=fp)
                if err is not None:
                    fp.write(str(err))

    def log_str(self, s, mode="a", newline=True):
        if self.log_filename != '':
            if newline and not s.endswith("\n"):
                s += "\n"
            with open(self.log_filename, mode) as fp:
                fp.write(s)

    @call_from_remote
    def target_constr(self):
        try:
            self.target = self.target_class(**self.target_kwargs)
            if hasattr(self.target, 'start'):
                self.target.start()
        except Exception as e:
            print("RPCProcess.target_constr: unable to start target!")
            print(e)
            self.log_error(mode='a')
            self.status.value = -1

    @call_from_remote
    def target_destr(self, ret_status, msg):
        pass

    @call_from_parent
    def start(self):
        super().start()
        self.target_proxy = ObjProxy(
            self.target_class,
            self._req_queue,
            self._resp_queue,
            log_filename=self.log_filename,
        )
        self.data_proxy = DataPipe(log_filename=self.log_filename)
        return self.target_proxy, self.data_proxy

    def check_run_condition(self):
        return self.status.value > 0

    def is_enabled(self):
        return self.status.value > 0

    @call_from_parent
    def stop(self):
        self.status.value = -1

    def __del__(self):
        """Stop the process when the object is destructed."""
        if self.status.value > 0:
            self.status.value = -1

    def is_cmd_present(self):
        return not self._req_queue.empty()

    @call_from_remote
    def loop_task(self):
        time.sleep(0.01)

    @call_from_remote
    def proc_rpc_command(self):
        try:
            cmd = self._req_queue.get_nowait()
        except queue.Empty:
            return

        self.log_str(f"Received command: {cmd}")

        if cmd is None:
            self.stop()
            return

        try:
            fn_name, cmd_args, cmd_kwargs = cmd
            fn = getattr(self.target, fn_name)
            self.log_str(f"Function: {fn}")
            fn_output = fn(*cmd_args, **cmd_kwargs)
            self._resp_queue.put(fn_output)
            self.log_str(f'Done with command: {fn_name}, output={fn_output}')
        except Exception as e:
            self.log_error(mode='a')
            self._resp_queue.put(e)

    @call_from_remote
    def run(self):
        self.log_str("RPCProcess.run")
        self.target_constr()

        try:
            while self.is_enabled():
                if not self.check_run_condition():
                    self.log_str("The target's termination condition was reached")
                    break

                if self.is_cmd_present():
                    self.proc_rpc_command()

                self.loop_task()

            self.log_str("RPCProcess.run: end of while loop")
            ret_status, msg = 0, ''
        except KeyboardInterrupt:
            ret_status, msg = 1, 'KeyboardInterrupt'
        except Exception:
            traceback.print_exc()
            err = io.StringIO()
            traceback.print_exc(file=err)
            err.seek(0)
            ret_status, msg = 1, err.read()
        finally:
            self.target_destr(ret_status, msg)
            self.status.value = -1