"""
Data Cache System for Visualization Components

This module provides a comprehensive multi-level caching system with memory and disk
persistence, configurable invalidation strategies, performance monitoring, and 
automatic cache management.
"""

import logging
import threading
import time
import hashlib
import pickle
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict

logger = logging.getLogger(__name__)


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    TIME_BASED = "time_based"  # TTL-based invalidation
    DEPENDENCY_BASED = "dependency_based"  # Dependency change invalidation
    VERSION_BASED = "version_based"  # Version change invalidation
    MANUAL = "manual"  # Manual invalidation only
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    key: str
    data: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[int] = None
    dependencies: Set[str] = field(default_factory=set)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds
    
    @property
    def age_seconds(self) -> float:
        """Get the age of the cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'size_bytes': self.size_bytes,
            'ttl_seconds': self.ttl_seconds,
            'dependencies': list(self.dependencies),
            'version': self.version,
            'metadata': self.metadata,
            'is_expired': self.is_expired,
            'age_seconds': self.age_seconds
        }


@dataclass
class CacheMetrics:
    """Cache performance and usage metrics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    avg_access_time_ms: float = 0.0
    last_cleanup_time: Optional[datetime] = None
    
    @property
    def hit_ratio(self) -> float:
        """Calculate cache hit ratio."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def total_requests(self) -> int:
        """Total cache requests."""
        return self.hits + self.misses
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'invalidations': self.invalidations,
            'size_bytes': self.size_bytes,
            'entry_count': self.entry_count,
            'memory_usage_mb': self.memory_usage_mb,
            'disk_usage_mb': self.disk_usage_mb,
            'avg_access_time_ms': self.avg_access_time_ms,
            'hit_ratio': self.hit_ratio,
            'total_requests': self.total_requests,
            'last_cleanup_time': self.last_cleanup_time.isoformat() if self.last_cleanup_time else None
        }


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""
    max_memory_size_mb: int = 100  # Maximum memory cache size
    max_disk_size_mb: int = 1000   # Maximum disk cache size
    max_entries: int = 1000        # Maximum number of entries
    default_ttl_seconds: int = 3600  # Default TTL (1 hour)
    eviction_policy: EvictionPolicy = EvictionPolicy.LRU
    invalidation_strategy: InvalidationStrategy = InvalidationStrategy.TIME_BASED
    cleanup_interval_seconds: int = 300  # Cleanup every 5 minutes
    enable_compression: bool = True
    enable_memory_cache: bool = True
    enable_disk_cache: bool = True
    disk_cache_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'max_memory_size_mb': self.max_memory_size_mb,
            'max_disk_size_mb': self.max_disk_size_mb,
            'max_entries': self.max_entries,
            'default_ttl_seconds': self.default_ttl_seconds,
            'eviction_policy': self.eviction_policy.value,
            'invalidation_strategy': self.invalidation_strategy.value,
            'cleanup_interval_seconds': self.cleanup_interval_seconds,
            'enable_compression': self.enable_compression,
            'enable_memory_cache': self.enable_memory_cache,
            'enable_disk_cache': self.enable_disk_cache,
            'disk_cache_dir': str(self.disk_cache_dir) if self.disk_cache_dir else None
        }


class MemoryCacheBackend:
    """In-memory cache backend with configurable eviction policies."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._current_size_bytes = 0
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get an entry from memory cache."""
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                return None
            
            # Update access info
            entry.touch()
            self._access_counts[key] += 1
            
            # Move to end for LRU
            if self.config.eviction_policy == EvictionPolicy.LRU:
                self._cache.move_to_end(key)
            
            return entry
    
    def put(self, entry: CacheEntry) -> bool:
        """Put an entry into memory cache."""
        with self._lock:
            # Calculate entry size
            entry.size_bytes = self._calculate_size(entry.data)
            
            # Check if we need to evict
            self._ensure_capacity(entry.size_bytes)
            
            # Remove existing entry if present
            if entry.key in self._cache:
                self._remove_entry(entry.key)
            
            # Add new entry
            self._cache[entry.key] = entry
            self._access_counts[entry.key] = 1
            self._current_size_bytes += entry.size_bytes
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove an entry from memory cache."""
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self) -> None:
        """Clear all entries from memory cache."""
        with self._lock:
            self._cache.clear()
            self._access_counts.clear()
            self._current_size_bytes = 0
    
    def size(self) -> int:
        """Get the number of entries in memory cache."""
        with self._lock:
            return len(self._cache)
    
    def keys(self) -> Set[str]:
        """Get all keys in memory cache."""
        with self._lock:
            return set(self._cache.keys())
    
    def get_size_bytes(self) -> int:
        """Get total size in bytes."""
        with self._lock:
            return self._current_size_bytes
    
    def _remove_entry(self, key: str) -> bool:
        """Remove an entry and update size tracking."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._access_counts.pop(key, None)
            self._current_size_bytes -= entry.size_bytes
            return True
        return False
    
    def _ensure_capacity(self, new_entry_size: int) -> None:
        """Ensure there's capacity for a new entry."""
        max_size_bytes = self.config.max_memory_size_mb * 1024 * 1024
        max_entries = self.config.max_entries
        
        # Evict by size
        while (self._current_size_bytes + new_entry_size > max_size_bytes or 
               len(self._cache) >= max_entries) and self._cache:
            self._evict_one_entry()
    
    def _evict_one_entry(self) -> None:
        """Evict one entry based on the eviction policy."""
        if not self._cache:
            return
        
        if self.config.eviction_policy == EvictionPolicy.LRU:
            key = next(iter(self._cache))
        elif self.config.eviction_policy == EvictionPolicy.LFU:
            key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        elif self.config.eviction_policy == EvictionPolicy.FIFO:
            key = next(iter(self._cache))
        elif self.config.eviction_policy == EvictionPolicy.TTL:
            expired_keys = [k for k, v in self._cache.items() if v.is_expired]
            if expired_keys:
                key = min(expired_keys, key=lambda k: self._cache[k].created_at)
            else:
                key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        else:
            key = next(iter(self._cache))
        
        self._remove_entry(key)
    
    def _calculate_size(self, data: Any) -> int:
        """Calculate approximate size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except:
            return len(str(data).encode('utf-8'))


class DataCache:
    """
    Multi-level data cache with configurable policies and monitoring.
    
    Features:
    - Memory caching with configurable eviction policies
    - Invalidation strategies (time-based, dependency-based, version-based)  
    - Performance monitoring and metrics
    - Thread-safe operations
    - Automatic cleanup and maintenance
    """
    
    def __init__(self, 
                 name: str = "default",
                 config: Optional[CacheConfig] = None):
        self.name = name
        self.config = config or CacheConfig()
        
        # Initialize backends
        self._memory_backend: Optional[MemoryCacheBackend] = None
        
        if self.config.enable_memory_cache:
            self._memory_backend = MemoryCacheBackend(self.config)
        
        # Metrics and monitoring
        self._metrics = CacheMetrics()
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._access_times: List[float] = []
        self._lock = threading.RLock()
        
        # Callbacks
        self._invalidation_callbacks: List[Callable[[str], None]] = []
        
        logger.info(f"DataCache '{name}' initialized with config: {self.config.to_dict()}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get data from cache."""
        start_time = time.time()
        
        with self._lock:
            entry = None
            
            # Try memory cache
            if self._memory_backend:
                entry = self._memory_backend.get(key)
            
            # Update metrics
            access_time = (time.time() - start_time) * 1000
            self._access_times.append(access_time)
            
            if entry is not None:
                self._metrics.hits += 1
                logger.debug(f"Cache hit for key: {key}")
                return entry.data
            else:
                self._metrics.misses += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
    
    def put(self, 
            key: str, 
            data: Any, 
            ttl_seconds: Optional[int] = None,
            dependencies: Optional[Set[str]] = None,
            version: str = "1.0",
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put data into cache."""
        with self._lock:
            now = datetime.now()
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                data=data,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl_seconds=ttl_seconds or self.config.default_ttl_seconds,
                dependencies=dependencies or set(),
                version=version,
                metadata=metadata or {}
            )
            
            success = True
            
            # Store in memory cache
            if self._memory_backend:
                memory_success = self._memory_backend.put(entry)
                success = success and memory_success
            
            # Track dependencies
            if dependencies:
                for dep in dependencies:
                    self._dependencies[dep].add(key)
            
            if success:
                logger.debug(f"Cache put successful for key: {key}")
            else:
                logger.warning(f"Cache put failed for key: {key}")
            
            return success
    
    def remove(self, key: str) -> bool:
        """Remove data from cache."""
        with self._lock:
            success = True
            
            # Remove from memory cache
            if self._memory_backend:
                memory_success = self._memory_backend.remove(key)
                success = success and memory_success
            
            # Clean up dependencies
            self._cleanup_dependencies_for_key(key)
            
            if success:
                self._metrics.invalidations += 1
                logger.debug(f"Cache remove successful for key: {key}")
            
            return success
    
    def invalidate_by_dependency(self, dependency_key: str) -> int:
        """Invalidate all cache entries that depend on the given key."""
        with self._lock:
            dependent_keys = self._dependencies.get(dependency_key, set()).copy()
            invalidated_count = 0
            
            for key in dependent_keys:
                if self.remove(key):
                    invalidated_count += 1
                    
                    # Notify callbacks
                    for callback in self._invalidation_callbacks:
                        try:
                            callback(key)
                        except Exception as e:
                            logger.error(f"Error in invalidation callback: {e}")
            
            # Clean up the dependency tracking
            if dependency_key in self._dependencies:
                del self._dependencies[dependency_key]
            
            logger.info(f"Invalidated {invalidated_count} cache entries for dependency: {dependency_key}")
            return invalidated_count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            if self._memory_backend:
                self._memory_backend.clear()
            
            self._dependencies.clear()
            self._access_times.clear()
            
            # Reset metrics
            self._metrics.hits = 0
            self._metrics.misses = 0
            self._metrics.evictions = 0
            self._metrics.invalidations = 0
            
            logger.info(f"Cache '{self.name}' cleared")
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        with self._lock:
            cleaned_count = 0
            
            # Get all keys from memory backend
            all_keys = set()
            if self._memory_backend:
                all_keys.update(self._memory_backend.keys())
            
            for key in all_keys:
                entry = None
                
                # Get entry to check expiration
                if self._memory_backend:
                    entry = self._memory_backend.get(key)
                
                if entry and entry.is_expired:
                    if self.remove(key):
                        cleaned_count += 1
            
            self._metrics.last_cleanup_time = datetime.now()
            logger.debug(f"Cleaned up {cleaned_count} expired cache entries")
            return cleaned_count
    
    def get_metrics(self) -> CacheMetrics:
        """Get current cache metrics."""
        with self._lock:
            # Calculate average access time
            if self._access_times:
                self._metrics.avg_access_time_ms = sum(self._access_times) / len(self._access_times)
                # Keep only recent access times to avoid memory growth
                if len(self._access_times) > 1000:
                    self._access_times = self._access_times[-500:]
            
            # Update size metrics
            if self._memory_backend:
                self._metrics.entry_count = self._memory_backend.size()
                self._metrics.memory_usage_mb = self._memory_backend.get_size_bytes() / (1024 * 1024)
            
            return self._metrics
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information."""
        with self._lock:
            info = {
                'name': self.name,
                'config': self.config.to_dict(),
                'metrics': self.get_metrics().to_dict(),
                'backends': {
                    'memory': {
                        'enabled': self._memory_backend is not None,
                        'size': self._memory_backend.size() if self._memory_backend else 0,
                        'size_bytes': self._memory_backend.get_size_bytes() if self._memory_backend else 0
                    }
                },
                'dependencies': {k: list(v) for k, v in self._dependencies.items()},
                'callback_count': len(self._invalidation_callbacks)
            }
            
            return info
    
    def add_invalidation_callback(self, callback: Callable[[str], None]) -> None:
        """Add a callback that will be called when entries are invalidated."""
        self._invalidation_callbacks.append(callback)
    
    def remove_invalidation_callback(self, callback: Callable[[str], None]) -> None:
        """Remove an invalidation callback."""
        if callback in self._invalidation_callbacks:
            self._invalidation_callbacks.remove(callback)
    
    def _cleanup_dependencies_for_key(self, key: str) -> None:
        """Clean up dependency tracking for a removed key."""
        # Remove key from all dependency sets
        for dependency_set in self._dependencies.values():
            dependency_set.discard(key)
        
        # Remove empty dependency entries
        empty_deps = [dep for dep, keys in self._dependencies.items() if not keys]
        for dep in empty_deps:
            del self._dependencies[dep]


class CacheManager:
    """Global cache manager for managing multiple named caches."""
    
    def __init__(self):
        self._caches: Dict[str, DataCache] = {}
        self._default_config = CacheConfig()
        self._lock = threading.RLock()
    
    def get_cache(self, name: str = "default", config: Optional[CacheConfig] = None) -> DataCache:
        """Get or create a named cache."""
        with self._lock:
            if name not in self._caches:
                cache_config = config or self._default_config
                self._caches[name] = DataCache(name, cache_config)
            
            return self._caches[name]
    
    def remove_cache(self, name: str) -> bool:
        """Remove a named cache."""
        with self._lock:
            if name in self._caches:
                self._caches.pop(name)
                return True
            return False
    
    def clear_all(self) -> None:
        """Clear all caches."""
        with self._lock:
            for cache in self._caches.values():
                cache.clear()
    
    def get_all_metrics(self) -> Dict[str, CacheMetrics]:
        """Get metrics for all caches."""
        with self._lock:
            return {name: cache.get_metrics() for name, cache in self._caches.items()}
    
    def get_cache_names(self) -> List[str]:
        """Get names of all active caches."""
        with self._lock:
            return list(self._caches.keys())


# Global cache manager instance
cache_manager = CacheManager() 