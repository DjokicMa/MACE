#!/usr/bin/env python3
"""
Queue Lock Manager for Race Condition Prevention
-----------------------------------------------
Provides distributed locking mechanisms to prevent race conditions when
multiple queue managers are triggered simultaneously.

Features:
- File-based distributed locking with timeout
- Process-level mutex for local synchronization
- Randomized backoff for collision reduction
- Lock cleanup on process termination

Author: Concurrency control system
"""

import os
import sys
import time
import fcntl
import signal
import atexit
import random
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Dict
import tempfile
import socket


class QueueLockManager:
    """
    Manages distributed locking for queue operations to prevent race conditions.
    
    Uses a combination of file-based locking (for distributed systems) and
    thread-based locking (for local concurrency).
    """
    
    def __init__(self, lock_dir: Optional[Path] = None, lock_timeout: int = 300):
        """
        Initialize the lock manager.
        
        Args:
            lock_dir: Directory for lock files (defaults to system temp)
            lock_timeout: Maximum time to hold a lock (seconds)
        """
        self.lock_dir = lock_dir or Path(tempfile.gettempdir()) / "crystal_queue_locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_timeout = lock_timeout
        self.held_locks = {}
        self.local_lock = threading.RLock()
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
        
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
        
    def _write_lock_info(self, lock_file: Path, fd: int):
        """Write lock information to the lock file."""
        lock_info = {
            'hostname': self.hostname,
            'pid': self.pid,
            'acquired_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=self.lock_timeout)).isoformat()
        }
        os.write(fd, f"{lock_info}\n".encode())
        os.fsync(fd)
        
    def _is_lock_expired(self, lock_file: Path) -> bool:
        """Check if a lock file has expired."""
        try:
            if not lock_file.exists():
                return True
                
            # Check file modification time
            mtime = datetime.fromtimestamp(lock_file.stat().st_mtime)
            if datetime.now() - mtime > timedelta(seconds=self.lock_timeout):
                return True
                
            # Try to read lock info
            try:
                with open(lock_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        import ast
                        lock_info = ast.literal_eval(content)
                        expires_at = datetime.fromisoformat(lock_info.get('expires_at', ''))
                        return datetime.now() > expires_at
            except Exception:
                pass
                
            return False
            
        except Exception:
            return True
            
    def acquire_lock(self, lock_name: str, timeout: int = 30, 
                    retry_interval: float = 0.1) -> bool:
        """
        Acquire a distributed lock with timeout and retry.
        
        Args:
            lock_name: Name of the lock to acquire
            timeout: Maximum time to wait for lock (seconds)
            retry_interval: Base interval between retries (seconds)
            
        Returns:
            True if lock acquired, False if timeout
        """
        with self.local_lock:
            if lock_name in self.held_locks:
                return True  # Already held
                
        lock_file = self._get_lock_file_path(lock_name)
        start_time = time.time()
        attempt = 0
        
        while time.time() - start_time < timeout:
            try:
                # Try to clean up expired lock
                if self._is_lock_expired(lock_file):
                    try:
                        lock_file.unlink(missing_ok=True)
                    except Exception:
                        pass
                        
                # Try to acquire lock
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                
                try:
                    # Successfully created lock file
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._write_lock_info(lock_file, fd)
                    
                    with self.local_lock:
                        self.held_locks[lock_name] = {
                            'fd': fd,
                            'file': lock_file,
                            'acquired_at': datetime.now()
                        }
                    
                    return True
                    
                except IOError:
                    # Could not get exclusive lock
                    os.close(fd)
                    lock_file.unlink(missing_ok=True)
                    
            except FileExistsError:
                # Lock file already exists
                pass
            except Exception as e:
                print(f"Error acquiring lock {lock_name}: {e}")
                
            # Randomized exponential backoff
            attempt += 1
            sleep_time = retry_interval * (2 ** min(attempt, 5)) + random.uniform(0, 0.1)
            time.sleep(min(sleep_time, 1.0))
            
        return False
        
    def release_lock(self, lock_name: str):
        """Release a held lock."""
        with self.local_lock:
            self._release_lock(lock_name)
            
    def _release_lock(self, lock_name: str):
        """Internal method to release a lock."""
        if lock_name not in self.held_locks:
            return
            
        lock_info = self.held_locks[lock_name]
        
        try:
            # Release file lock
            fcntl.flock(lock_info['fd'], fcntl.LOCK_UN)
            os.close(lock_info['fd'])
            
            # Remove lock file
            lock_info['file'].unlink(missing_ok=True)
            
        except Exception as e:
            print(f"Error releasing lock {lock_name}: {e}")
        finally:
            del self.held_locks[lock_name]
            
    def with_lock(self, lock_name: str, func: Callable, *args, 
                 timeout: int = 30, **kwargs) -> Any:
        """
        Execute a function with a distributed lock.
        
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
            # Get held locks
            for lock_name, lock_info in self.held_locks.items():
                status['held_locks'].append({
                    'name': lock_name,
                    'acquired_at': lock_info['acquired_at'].isoformat(),
                    'file': str(lock_info['file'])
                })
                
        # Get all lock files
        try:
            for lock_file in self.lock_dir.glob("*.lock"):
                lock_name = lock_file.stem
                expired = self._is_lock_expired(lock_file)
                
                lock_status = {
                    'name': lock_name,
                    'file': str(lock_file),
                    'expired': expired,
                    'held_by_us': lock_name in self.held_locks
                }
                
                try:
                    mtime = datetime.fromtimestamp(lock_file.stat().st_mtime)
                    lock_status['modified'] = mtime.isoformat()
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