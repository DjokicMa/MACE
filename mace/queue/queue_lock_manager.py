#!/usr/bin/env python3
"""
Queue Lock Manager for Race Condition Prevention
-----------------------------------------------
Mutual exclusion for the callbacks and sweeps that change the queue state
(the database rows, submissions, recovery resubmissions), which several
jobs can trigger at the same moment.

The lock is an OS lock (a POSIX record lock, fcntl) on a lock file that is
never deleted. It is held for exactly as long as its holder runs the
critical section and goes away with the holder's process, so it has no
expiry: a long sweep is never broken into, and a late release can never
remove another holder's lock. POSIX record locks are coherent across the
nodes of a cluster on NFS (lockd / NFSv4) and on GPFS. On a filesystem
that has no working locks (lockf fails with ENOLCK / ENOSYS / EOPNOTSUPP),
or when MACE_QUEUE_LOCK=token is set (e.g. Lustre mounted with  # british-ok: Lustre is a file system
'localflock', whose locks are only node-local), an owner-token file with a
heartbeat is used instead: its holder refreshes it while it runs, a waiter
takes it over only when the heartbeat stopped for lock_timeout seconds,
and a release deletes it only while it still carries the holder's token.
"""

import os
import sys
import time
import errno
import fcntl
import signal
import atexit
import random
import threading
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, Any, Dict
import tempfile
import socket


# The one lock every queue-changing callback mode and sweep takes: the
# completion callbacks and `mace manager` sweeps both resubmit through
# recovery, so they must exclude each other. (Before 1.1.5 the lock was
# named per mode, queue_manager_<mode>.)
QUEUE_STATE_LOCK = "queue_state"

# lockf errors that mean "this filesystem has no working locks"
_NO_LOCKS = {errno.ENOLCK, errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP}


class QueueLockManager:
    """
    Manages locking for queue operations to prevent race conditions.

    An OS lock on a persistent lock file (see the module docstring), plus a
    thread lock for the threads of one process.
    """

    def __init__(self, lock_dir: Optional[Path] = None, lock_timeout: int = 300):
        """
        Initialize the lock manager.

        Args:
            lock_dir: Directory for lock files (defaults to system temp)
            lock_timeout: Seconds without a heartbeat after which a token lock
                (the fallback for filesystems without locks) counts as
                abandoned. An OS lock never expires: it is held until its
                holder releases it or exits.
        """
        self.lock_dir = lock_dir or Path(tempfile.gettempdir()) / "crystal_queue_locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_timeout = lock_timeout
        self.held_locks = {}
        self.local_lock = threading.RLock()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        self.use_token = (os.environ.get('MACE_QUEUE_LOCK') or '').strip().lower() == 'token'

        # Register cleanup on exit
        atexit.register(self._cleanup_locks)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals by cleaning up locks."""
        self._cleanup_locks()
        sys.exit(0)

    def _cleanup_locks(self):
        """Release all held locks on process termination."""
        with self.local_lock:
            for lock_name, lock_info in list(self.held_locks.items()):
                try:
                    self._release_lock(lock_name)
                except Exception:
                    pass

    def _get_lock_file_path(self, lock_name: str) -> Path:
        """Get the path for a lock file."""
        return self.lock_dir / f"{lock_name}.lock"

    def _get_token_file_path(self, lock_name: str) -> Path:
        """The owner-token file of the fallback lock."""
        return self.lock_dir / f"{lock_name}.token"

    def _holder_line(self, token: str = '') -> str:
        return (f"{{'hostname': {self.hostname!r}, 'pid': {self.pid}, "
                f"'acquired_at': {datetime.now().isoformat()!r}, 'token': {token!r}}}\n")

    # -- OS lock -----------------------------------------------------------

    def _try_os_lock(self, lock_name: str) -> Optional[bool]:
        """One non-blocking try. True: held; False: busy; None: this
        filesystem has no working locks."""
        lock_file = self._get_lock_file_path(lock_name)
        # 0666 less the umask: in a folder a group shares, the file another
        # member created must open for everyone (0644 locked them all out).
        try:
            fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o666)
        except PermissionError:
            # an older MACE (or a strict umask) made it read-only for this
            # user: the owner-token lock still excludes other sweeps
            print(f"Warning: cannot write {lock_file} (another user's); using the owner-token lock, "
                  f"which a sweep holding that file does not see - its owner can `chmod g+w` it")
            return None
        try:
            fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EACCES, errno.EAGAIN):
                return False
            if e.errno in _NO_LOCKS:
                return None
            raise
        try:
            # who holds it, for a person looking at the file; never read back
            os.ftruncate(fd, 0)
            os.write(fd, self._holder_line().encode())
        except OSError:
            pass
        self.held_locks[lock_name] = {'fd': fd, 'file': lock_file, 'kind': 'os',
                                      'acquired_at': datetime.now()}
        return True

    # -- owner-token fallback ---------------------------------------------

    @staticmethod
    def _read(path: Path) -> Optional[str]:
        try:
            return path.read_text()
        except OSError:
            return None

    def _token_abandoned(self, token_file: Path, content: str) -> bool:
        try:
            age = time.time() - token_file.stat().st_mtime
        except OSError:
            return False
        if age > self.lock_timeout:
            return True
        # a holder on this host whose process is gone
        try:
            import ast
            info = ast.literal_eval(content.strip())
            if info.get('hostname') == self.hostname and int(info.get('pid')) != self.pid:
                os.kill(int(info['pid']), 0)
        except ProcessLookupError:
            return True
        except Exception:
            pass
        return False

    def _try_token_lock(self, lock_name: str) -> bool:
        token_file = self._get_token_file_path(lock_name)
        token = f"{self.hostname}:{self.pid}:{uuid.uuid4().hex}"
        try:
            fd = os.open(str(token_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            content = self._read(token_file)
            if content is not None and self._token_abandoned(token_file, content):
                # Take the abandoned file out of the way by an atomic rename;
                # if what moved is not what was judged abandoned (it was
                # replaced in between), put it back.
                aside = token_file.with_name(f"{token_file.name}.{uuid.uuid4().hex}")
                try:
                    os.rename(token_file, aside)
                except OSError:
                    return False
                if self._read(aside) != content:
                    try:
                        os.link(aside, token_file)
                    except OSError:
                        pass
                try:
                    os.unlink(aside)
                except OSError:
                    pass
            return False
        try:
            os.write(fd, self._holder_line(token).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(token_file, token, stop), daemon=True)
        self.held_locks[lock_name] = {'file': token_file, 'kind': 'token', 'token': token,
                                      'stop': stop, 'acquired_at': datetime.now()}
        beat.start()
        return True

    def _heartbeat(self, token_file: Path, token: str, stop: threading.Event):
        interval = max(min(self.lock_timeout / 10.0, 30.0), 0.2)
        while not stop.wait(interval):
            content = self._read(token_file)
            if content is None or token not in content:
                return
            try:
                os.utime(token_file)
            except OSError:
                pass

    # -- public ------------------------------------------------------------

    def _try_acquire(self, lock_name: str) -> bool:
        with self.local_lock:
            if lock_name in self.held_locks:
                return True
            if not self.use_token:
                got = self._try_os_lock(lock_name)
                if got is not None:
                    return got
                self.use_token = True
            return self._try_token_lock(lock_name)

    def acquire_lock(self, lock_name: str, timeout: int = 30,
                    retry_interval: float = 0.1) -> bool:
        """
        Acquire the lock, waiting up to `timeout` seconds.

        Args:
            lock_name: Name of the lock to acquire
            timeout: Maximum time to wait for lock (seconds)
            retry_interval: Base interval between retries (seconds)

        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        attempt = 0
        while True:
            try:
                if self._try_acquire(lock_name):
                    return True
            except Exception as e:
                print(f"Error acquiring lock {lock_name}: {e}")
            remaining = timeout - (time.time() - start_time)
            if remaining <= 0:
                return False
            # Randomized exponential backoff
            attempt += 1
            sleep_time = retry_interval * (2 ** min(attempt, 5)) + random.uniform(0, 0.1)
            time.sleep(max(min(sleep_time, 1.0, remaining), 0))

    def release_lock(self, lock_name: str):
        """Release a held lock."""
        with self.local_lock:
            self._release_lock(lock_name)

    def _release_lock(self, lock_name: str):
        """Internal method to release a lock. The OS lock file is never
        deleted; a token file only while it is still this holder's."""
        if lock_name not in self.held_locks:
            return

        lock_info = self.held_locks[lock_name]

        try:
            if lock_info.get('kind') == 'token':
                lock_info['stop'].set()
                content = self._read(lock_info['file'])
                if content is not None and lock_info['token'] in content:
                    lock_info['file'].unlink(missing_ok=True)
            else:
                try:
                    os.ftruncate(lock_info['fd'], 0)
                except OSError:
                    pass
                fcntl.lockf(lock_info['fd'], fcntl.LOCK_UN)
                os.close(lock_info['fd'])
        except Exception as e:
            print(f"Error releasing lock {lock_name}: {e}")
        finally:
            del self.held_locks[lock_name]

    def with_lock(self, lock_name: str, func: Callable, *args,
                 timeout: int = 30, **kwargs) -> Any:
        """
        Execute a function with the lock held.

        Args:
            lock_name: Name of the lock
            func: Function to execute
            timeout: Lock acquisition timeout
            *args, **kwargs: Arguments for the function

        Returns:
            Result of the function

        Raises:
            TimeoutError: If lock cannot be acquired
        """
        if not self.acquire_lock(lock_name, timeout):
            raise TimeoutError(f"Could not acquire lock '{lock_name}' within {timeout} seconds")

        try:
            return func(*args, **kwargs)
        finally:
            self.release_lock(lock_name)

    def get_lock_status(self) -> Dict[str, Any]:
        """Get status of all locks in the system."""
        status = {
            'held_locks': [],
            'all_locks': []
        }

        with self.local_lock:
            for lock_name, lock_info in self.held_locks.items():
                status['held_locks'].append({
                    'name': lock_name,
                    'acquired_at': lock_info['acquired_at'].isoformat(),
                    'file': str(lock_info['file'])
                })

            # Lock files only (never opened: closing any descriptor of a file
            # drops this process's POSIX lock on it)
            try:
                for lock_file in sorted(self.lock_dir.glob("*.lock")) + sorted(self.lock_dir.glob("*.token")):
                    lock_name = lock_file.stem
                    lock_status = {
                        'name': lock_name,
                        'file': str(lock_file),
                        'held_by_us': lock_name in self.held_locks
                    }
                    try:
                        lock_status['modified'] = datetime.fromtimestamp(lock_file.stat().st_mtime).isoformat()
                    except Exception:
                        pass
                    status['all_locks'].append(lock_status)
            except Exception as e:
                status['error'] = str(e)

        return status


class CallbackThrottler:
    """
    Throttles and randomizes callbacks to reduce simultaneous executions.
    """
    
    def __init__(self, min_delay: float = 0.5, max_delay: float = 2.0):
        """
        Initialize the callback throttler.
        
        Args:
            min_delay: Minimum delay before callback (seconds)
            max_delay: Maximum delay before callback (seconds)
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_callback_time = {}
        self.lock = threading.Lock()
        
    def throttle(self, callback_type: str = "default"):
        """
        Apply throttling with randomized delay.
        
        Args:
            callback_type: Type of callback for separate throttling
        """
        with self.lock:
            now = time.time()
            last_time = self.last_callback_time.get(callback_type, 0)
            
            # Calculate time since last callback
            time_since_last = now - last_time
            
            # If recent callback, add extra delay
            if time_since_last < self.min_delay:
                extra_delay = self.min_delay - time_since_last
            else:
                extra_delay = 0
                
        # Randomized delay to spread out simultaneous callbacks
        delay = random.uniform(self.min_delay, self.max_delay) + extra_delay
        time.sleep(delay)
        
        with self.lock:
            self.last_callback_time[callback_type] = time.time()


# Example usage in callback
def safe_queue_callback(queue_manager, lock_manager: QueueLockManager, 
                       throttler: CallbackThrottler):
    """
    Example of a safe callback with locking and throttling.
    """
    # Apply randomized throttling
    throttler.throttle("queue_callback")
    
    # Acquire distributed lock
    try:
        def run_callback():
            # Your actual callback logic here
            queue_manager.process_completed_jobs()
            queue_manager.submit_new_jobs()
            
        lock_manager.with_lock("queue_manager_main", run_callback, timeout=60)
        
    except TimeoutError:
        print("Could not acquire queue lock - another instance may be running")
        return
    except Exception as e:
        print(f"Error in queue callback: {e}")
        raise