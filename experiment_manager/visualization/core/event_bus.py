"""
Event Bus System for Visualization Components

This module provides event-driven communication between visualization components,
enabling loose coupling and extensible architecture. The event bus supports
synchronous and asynchronous event handling, event filtering, and prioritization.
"""

import asyncio
import logging
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import traceback

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels for ordering event processing."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """Standard event types for visualization lifecycle."""
    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    
    # Plugin events
    PLUGIN_REGISTERED = "plugin.registered"
    PLUGIN_UNREGISTERED = "plugin.unregistered"
    PLUGIN_INITIALIZED = "plugin.initialized"
    PLUGIN_ERROR = "plugin.error"
    
    # Data events
    DATA_LOADING = "data.loading"
    DATA_LOADED = "data.loaded"
    DATA_PROCESSING = "data.processing"
    DATA_PROCESSED = "data.processed"
    DATA_ERROR = "data.error"
    
    # Plot events
    PLOT_CREATING = "plot.creating"
    PLOT_CREATED = "plot.created"
    PLOT_RENDERING = "plot.rendering"
    PLOT_RENDERED = "plot.rendered"
    PLOT_UPDATING = "plot.updating"
    PLOT_UPDATED = "plot.updated"
    PLOT_ERROR = "plot.error"
    
    # Export events
    EXPORT_STARTING = "export.starting"
    EXPORT_COMPLETED = "export.completed"
    EXPORT_ERROR = "export.error"
    
    # Theme events
    THEME_CHANGED = "theme.changed"
    THEME_APPLIED = "theme.applied"
    
    # Performance events
    PERFORMANCE_METRIC = "performance.metric"
    MEMORY_WARNING = "memory.warning"
    
    # User interaction events
    USER_INTERACTION = "user.interaction"
    CONFIG_CHANGED = "config.changed"


@dataclass
class Event:
    """
    Event object containing event data and metadata.
    
    Represents a single event in the system with its type, data, metadata,
    and processing information.
    """
    event_type: Union[EventType, str]
    data: Any = None
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        """Ensure event_type is properly formatted."""
        if isinstance(self.event_type, EventType):
            self.event_type = self.event_type.value
    
    @property
    def event_name(self) -> str:
        """Get the event type as string."""
        return self.event_type
    
    def has_tag(self, tag: str) -> bool:
        """Check if event has a specific tag."""
        return tag in self.tags
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the event."""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the event."""
        self.tags.discard(tag)


class EventFilter:
    """
    Filter for selecting events based on criteria.
    
    Allows filtering events by type, source, tags, and custom predicates.
    """
    
    def __init__(self, 
                 event_types: Optional[Union[str, List[str]]] = None,
                 sources: Optional[Union[str, List[str]]] = None,
                 tags: Optional[Union[str, List[str]]] = None,
                 priority_min: Optional[EventPriority] = None,
                 priority_max: Optional[EventPriority] = None,
                 predicate: Optional[Callable[[Event], bool]] = None):
        """
        Initialize event filter.
        
        Args:
            event_types: Event types to match (string or list)
            sources: Event sources to match (string or list)
            tags: Tags that events must have (string or list)
            priority_min: Minimum priority level
            priority_max: Maximum priority level
            predicate: Custom filter function
        """
        self.event_types = self._ensure_list(event_types)
        self.sources = self._ensure_list(sources)
        self.tags = self._ensure_list(tags)
        self.priority_min = priority_min
        self.priority_max = priority_max
        self.predicate = predicate
    
    def matches(self, event: Event) -> bool:
        """
        Check if event matches this filter.
        
        Args:
            event: Event to check
            
        Returns:
            True if event matches all filter criteria
        """
        # Check event types
        if self.event_types and event.event_name not in self.event_types:
            return False
        
        # Check sources
        if self.sources and event.source not in self.sources:
            return False
        
        # Check tags
        if self.tags and not all(event.has_tag(tag) for tag in self.tags):
            return False
        
        # Check priority range
        if self.priority_min and event.priority.value < self.priority_min.value:
            return False
        
        if self.priority_max and event.priority.value > self.priority_max.value:
            return False
        
        # Check custom predicate
        if self.predicate and not self.predicate(event):
            return False
        
        return True
    
    def _ensure_list(self, value: Optional[Union[str, List[str]]]) -> Optional[List[str]]:
        """Convert string or None to list or None."""
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)


class EventHandler(ABC):
    """
    Abstract base class for event handlers.
    
    Event handlers can be synchronous or asynchronous and may filter
    which events they want to handle.
    """
    
    @property
    def handler_name(self) -> str:
        """Name of the handler for identification."""
        return self.__class__.__name__
    
    @property
    def event_filter(self) -> Optional[EventFilter]:
        """Filter for events this handler is interested in."""
        return None
    
    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """
        Handle an event.
        
        Args:
            event: Event to handle
        """
        pass
    
    def can_handle(self, event: Event) -> bool:
        """
        Check if this handler can handle the given event.
        
        Args:
            event: Event to check
            
        Returns:
            True if handler can handle the event
        """
        event_filter = self.event_filter
        return event_filter is None or event_filter.matches(event)


class EventSubscription:
    """
    Represents a subscription to events.
    
    Tracks the subscriber, handler function, filter, and metadata.
    """
    
    def __init__(self, 
                 subscriber: Any,
                 handler: Union[Callable[[Event], None], Callable[[Event], Awaitable[None]]],
                 event_filter: Optional[EventFilter] = None,
                 weak_ref: bool = True,
                 priority: int = 0):
        """
        Initialize event subscription.
        
        Args:
            subscriber: Object that owns this subscription
            handler: Function to call when event occurs
            event_filter: Filter for events to receive
            weak_ref: Whether to use weak reference to subscriber
            priority: Handler priority (higher = called first)
        """
        self.handler = handler
        self.event_filter = event_filter
        self.priority = priority
        self.is_async = asyncio.iscoroutinefunction(handler)
        
        # Use weak reference if requested and object supports it
        if weak_ref and hasattr(subscriber, '__weakref__'):
            self._subscriber_ref = weakref.ref(subscriber)
            self._use_weak_ref = True
        else:
            self._subscriber_ref = subscriber
            self._use_weak_ref = False
    
    @property
    def subscriber(self) -> Optional[Any]:
        """Get the subscriber object."""
        if self._use_weak_ref:
            return self._subscriber_ref()
        return self._subscriber_ref
    
    @property
    def is_valid(self) -> bool:
        """Check if subscription is still valid (subscriber exists)."""
        return self.subscriber is not None
    
    def can_handle(self, event: Event) -> bool:
        """Check if this subscription should handle the event."""
        if not self.is_valid:
            return False
        
        return self.event_filter is None or self.event_filter.matches(event)


class EventBusError(Exception):
    """Base exception for event bus errors."""
    pass


class EventHandlingError(EventBusError):
    """Raised when event handling fails."""
    pass


class EventBus:
    """
    Event bus for decoupled communication between visualization components.
    
    The event bus provides:
    - Publish/subscribe pattern for loose coupling
    - Synchronous and asynchronous event handling
    - Event filtering and prioritization
    - Weak references to prevent memory leaks
    - Thread-safe operation
    - Event history and debugging support
    
    Example:
        ```python
        bus = EventBus()
        
        # Subscribe to events
        def on_data_loaded(event):
            print(f"Data loaded: {event.data}")
        
        bus.subscribe(
            subscriber=self,
            handler=on_data_loaded,
            event_filter=EventFilter(event_types=[EventType.DATA_LOADED])
        )
        
        # Publish event
        bus.publish(Event(
            event_type=EventType.DATA_LOADED,
            data={"rows": 1000},
            source="data_loader"
        ))
        ```
    """
    
    def __init__(self, 
                 max_history: int = 1000,
                 thread_pool_size: int = 4,
                 enable_debugging: bool = False):
        """
        Initialize the event bus.
        
        Args:
            max_history: Maximum number of events to keep in history
            thread_pool_size: Size of thread pool for sync handlers
            enable_debugging: Whether to enable detailed debugging
        """
        self._subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self._global_subscriptions: List[EventSubscription] = []
        self._event_history: deque = deque(maxlen=max_history)
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        self._enable_debugging = enable_debugging
        self._stats = {
            'events_published': 0,
            'events_handled': 0,
            'handler_errors': 0,
            'subscriptions_created': 0,
            'subscriptions_removed': 0
        }
        
        logger.info(f"EventBus initialized with history={max_history}, threads={thread_pool_size}")
    
    def subscribe(self, 
                  subscriber: Any,
                  handler: Union[Callable[[Event], None], Callable[[Event], Awaitable[None]]],
                  event_filter: Optional[EventFilter] = None,
                  weak_ref: bool = True,
                  priority: int = 0) -> str:
        """
        Subscribe to events.
        
        Args:
            subscriber: Object that owns this subscription
            handler: Function to call when event occurs
            event_filter: Filter for events to receive (None = all events)
            weak_ref: Whether to use weak reference to subscriber
            priority: Handler priority (higher = called first)
            
        Returns:
            Subscription ID for unsubscribing
        """
        subscription = EventSubscription(
            subscriber=subscriber,
            handler=handler,
            event_filter=event_filter,
            weak_ref=weak_ref,
            priority=priority
        )
        
        subscription_id = f"{id(subscription)}"
        
        with self._lock:
            # If filter is for specific event types, add to specific lists
            if event_filter and event_filter.event_types:
                for event_type in event_filter.event_types:
                    self._subscriptions[event_type].append(subscription)
                    # Sort by priority (higher first)
                    self._subscriptions[event_type].sort(
                        key=lambda s: s.priority, reverse=True
                    )
            else:
                # Global subscription (receives all events)
                self._global_subscriptions.append(subscription)
                self._global_subscriptions.sort(
                    key=lambda s: s.priority, reverse=True
                )
            
            self._stats['subscriptions_created'] += 1
        
        if self._enable_debugging:
            logger.debug(f"Subscription created: {subscription_id} for {subscriber}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID returned from subscribe()
            
        Returns:
            True if subscription was found and removed
        """
        with self._lock:
            # Check global subscriptions
            for i, sub in enumerate(self._global_subscriptions):
                if f"{id(sub)}" == subscription_id:
                    del self._global_subscriptions[i]
                    self._stats['subscriptions_removed'] += 1
                    return True
            
            # Check event-specific subscriptions
            for event_type, subscriptions in self._subscriptions.items():
                for i, sub in enumerate(subscriptions):
                    if f"{id(sub)}" == subscription_id:
                        del subscriptions[i]
                        self._stats['subscriptions_removed'] += 1
                        return True
        
        return False
    
    def unsubscribe_all(self, subscriber: Any) -> int:
        """
        Remove all subscriptions for a subscriber.
        
        Args:
            subscriber: Subscriber object
            
        Returns:
            Number of subscriptions removed
        """
        removed_count = 0
        
        with self._lock:
            # Remove from global subscriptions
            self._global_subscriptions = [
                sub for sub in self._global_subscriptions
                if sub.subscriber != subscriber
            ]
            
            # Remove from event-specific subscriptions
            for event_type, subscriptions in self._subscriptions.items():
                before_count = len(subscriptions)
                self._subscriptions[event_type] = [
                    sub for sub in subscriptions
                    if sub.subscriber != subscriber
                ]
                removed_count += before_count - len(self._subscriptions[event_type])
        
        self._stats['subscriptions_removed'] += removed_count
        
        if self._enable_debugging:
            logger.debug(f"Removed {removed_count} subscriptions for {subscriber}")
        
        return removed_count
    
    def publish(self, event: Event, async_mode: bool = True) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
            async_mode: Whether to handle async handlers asynchronously
        """
        if self._enable_debugging:
            logger.debug(f"Publishing event: {event.event_name} from {event.source}")
        
        with self._lock:
            self._event_history.append(event)
            self._stats['events_published'] += 1
        
        # Get relevant subscriptions
        subscriptions = self._get_subscriptions_for_event(event)
        
        if not subscriptions:
            if self._enable_debugging:
                logger.debug(f"No subscribers for event: {event.event_name}")
            return
        
        # Handle subscriptions
        if async_mode:
            # Run in asyncio event loop if available
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._handle_subscriptions_async(event, subscriptions))
                else:
                    loop.run_until_complete(self._handle_subscriptions_async(event, subscriptions))
            except RuntimeError:
                # No event loop, handle synchronously
                asyncio.run(self._handle_subscriptions_async(event, subscriptions))
        else:
            # Handle synchronously
            self._handle_subscriptions_sync(event, subscriptions)
    
    async def publish_async(self, event: Event) -> None:
        """
        Async version of publish for use in async contexts.
        
        Args:
            event: Event to publish
        """
        if self._enable_debugging:
            logger.debug(f"Publishing event async: {event.event_name} from {event.source}")
        
        with self._lock:
            self._event_history.append(event)
            self._stats['events_published'] += 1
        
        subscriptions = self._get_subscriptions_for_event(event)
        
        if subscriptions:
            await self._handle_subscriptions_async(event, subscriptions)
    
    def get_event_history(self, 
                         limit: Optional[int] = None,
                         event_filter: Optional[EventFilter] = None) -> List[Event]:
        """
        Get event history with optional filtering.
        
        Args:
            limit: Maximum number of events to return
            event_filter: Filter to apply to events
            
        Returns:
            List of events from history
        """
        with self._lock:
            events = list(self._event_history)
        
        # Apply filter if provided
        if event_filter:
            events = [event for event in events if event_filter.matches(event)]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get event bus statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats['active_subscriptions'] = (
                len(self._global_subscriptions) +
                sum(len(subs) for subs in self._subscriptions.values())
            )
            stats['event_types_subscribed'] = len(self._subscriptions)
            stats['history_size'] = len(self._event_history)
        
        return stats
    
    def clear_history(self) -> None:
        """Clear the event history."""
        with self._lock:
            self._event_history.clear()
        
        logger.info("Event history cleared")
    
    def shutdown(self) -> None:
        """
        Shutdown the event bus and cleanup resources.
        """
        logger.info("Shutting down EventBus...")
        
        # Publish shutdown event
        shutdown_event = Event(
            event_type=EventType.SYSTEM_SHUTDOWN,
            source="event_bus"
        )
        self.publish(shutdown_event, async_mode=False)
        
        # Cleanup
        with self._lock:
            self._subscriptions.clear()
            self._global_subscriptions.clear()
            self._event_history.clear()
        
        # Shutdown thread pool
        self._thread_pool.shutdown(wait=True)
        
        logger.info("EventBus shutdown complete")
    
    def _get_subscriptions_for_event(self, event: Event) -> List[EventSubscription]:
        """Get all subscriptions that should handle this event."""
        subscriptions = []
        
        with self._lock:
            # Add global subscriptions
            subscriptions.extend([
                sub for sub in self._global_subscriptions
                if sub.can_handle(event)
            ])
            
            # Add event-specific subscriptions
            if event.event_name in self._subscriptions:
                subscriptions.extend([
                    sub for sub in self._subscriptions[event.event_name]
                    if sub.can_handle(event)
                ])
        
        # Remove invalid subscriptions (dead weak references)
        valid_subscriptions = []
        for sub in subscriptions:
            if sub.is_valid:
                valid_subscriptions.append(sub)
            else:
                # Remove dead subscription
                self._remove_dead_subscription(sub)
        
        # Sort by priority (higher first)
        valid_subscriptions.sort(key=lambda s: s.priority, reverse=True)
        
        return valid_subscriptions
    
    def _remove_dead_subscription(self, dead_subscription: EventSubscription) -> None:
        """Remove a dead subscription from all lists."""
        # Remove from global subscriptions
        self._global_subscriptions = [
            sub for sub in self._global_subscriptions
            if sub != dead_subscription
        ]
        
        # Remove from event-specific subscriptions
        for event_type, subscriptions in self._subscriptions.items():
            self._subscriptions[event_type] = [
                sub for sub in subscriptions
                if sub != dead_subscription
            ]
    
    async def _handle_subscriptions_async(self, 
                                        event: Event, 
                                        subscriptions: List[EventSubscription]) -> None:
        """Handle subscriptions asynchronously."""
        tasks = []
        
        for subscription in subscriptions:
            if subscription.is_async:
                # Async handler
                task = asyncio.create_task(
                    self._call_async_handler(subscription, event)
                )
                tasks.append(task)
            else:
                # Sync handler - run in thread pool
                task = asyncio.create_task(
                    asyncio.get_event_loop().run_in_executor(
                        self._thread_pool,
                        self._call_sync_handler,
                        subscription,
                        event
                    )
                )
                tasks.append(task)
        
        # Wait for all handlers to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _handle_subscriptions_sync(self, 
                                 event: Event, 
                                 subscriptions: List[EventSubscription]) -> None:
        """Handle subscriptions synchronously."""
        for subscription in subscriptions:
            if subscription.is_async:
                # Run async handler in new event loop
                try:
                    asyncio.run(self._call_async_handler(subscription, event))
                except Exception as e:
                    self._handle_error(subscription, event, e)
            else:
                # Call sync handler directly
                self._call_sync_handler(subscription, event)
    
    async def _call_async_handler(self, 
                                subscription: EventSubscription, 
                                event: Event) -> None:
        """Call an async event handler."""
        try:
            await subscription.handler(event)
            self._stats['events_handled'] += 1
        except Exception as e:
            self._handle_error(subscription, event, e)
    
    def _call_sync_handler(self, 
                          subscription: EventSubscription, 
                          event: Event) -> None:
        """Call a sync event handler."""
        try:
            subscription.handler(event)
            self._stats['events_handled'] += 1
        except Exception as e:
            self._handle_error(subscription, event, e)
    
    def _handle_error(self, 
                     subscription: EventSubscription, 
                     event: Event, 
                     error: Exception) -> None:
        """Handle errors in event handlers."""
        self._stats['handler_errors'] += 1
        
        error_msg = (
            f"Error in event handler {subscription.handler} "
            f"for event {event.event_name}: {error}"
        )
        
        logger.error(error_msg)
        
        if self._enable_debugging:
            logger.error(f"Handler error traceback:\n{traceback.format_exc()}")
        
        # Publish error event (but don't let it cause infinite loops)
        try:
            error_event = Event(
                event_type=EventType.PLUGIN_ERROR,
                data={
                    'original_event': event,
                    'handler': str(subscription.handler),
                    'error': str(error),
                    'traceback': traceback.format_exc()
                },
                source="event_bus",
                priority=EventPriority.HIGH
            )
            
            # Publish synchronously to avoid recursion
            self._handle_subscriptions_sync(error_event, 
                                          self._get_subscriptions_for_event(error_event))
        except Exception:
            # Don't let error handling cause more errors
            pass 