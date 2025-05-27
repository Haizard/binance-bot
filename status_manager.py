import threading
import time
from typing import Dict, Any
import json
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class StatusManager:
    _instance = None
    _lock = threading.Lock()
    _redis_key = 'tradingbot:status'

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(StatusManager, cls).__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self.statuses: Dict[str, Dict[str, Any]] = {}
        self.status_lock = threading.Lock()
        self.log_file = 'status_log.txt'
        self.redis = None
        if REDIS_AVAILABLE:
            try:
                                self.redis = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)
                
            except Exception as e:
                print(f"[StatusManager] WARNING: Redis unavailable ({e}), falling back to in-memory status.")
                self.redis = None
        else:
            print("[StatusManager] WARNING: redis-py not installed, falling back to in-memory status.")

    def update(self, agent: str, status: Dict[str, Any]):
        status['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        if self.redis:
            try:
                self.redis.hset(self._redis_key, agent, json.dumps(status))
            except Exception as e:
                print(f"[StatusManager] WARNING: Redis update failed ({e}), using in-memory fallback.")
                with self.status_lock:
                    self.statuses[agent] = status
        else:
            with self.status_lock:
                self.statuses[agent] = status
        self._log_status(agent, status)

    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        if self.redis:
            try:
                all_status = self.redis.hgetall(self._redis_key)
                return {k: json.loads(v) for k, v in all_status.items()}
            except Exception as e:
                print(f"[StatusManager] WARNING: Redis read failed ({e}), using in-memory fallback.")
                with self.status_lock:
                    return dict(self.statuses)
        else:
            with self.status_lock:
                return dict(self.statuses)

    def _log_status(self, agent: str, status: Dict[str, Any]):
        with open(self.log_file, 'a') as f:
            f.write(f"[{status['timestamp']}] {agent}: {status}\n") 