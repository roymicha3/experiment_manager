import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Any



class BaseLogger(ABC):
    """
    Abstract base class for all loggers.
    Provides standard logging methods that delegate to the underlying logger.
    """
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self._setup_handler()

    @abstractmethod
    def _setup_handler(self) -> None:
        """Setup the specific handler for the logger implementation."""
        pass

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)


class FileLogger(BaseLogger):
    """
    Logger implementation that writes to a file.
    """
    def __init__(self, name: str, 
                 log_dir: str,
                 filename: Optional[str] = None,
                 level: str = "INFO"):
        
        if not os.path.exists(log_dir):
            raise ValueError(f"Log directory {log_dir} does not exist")
        
        self.log_dir = log_dir
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.log"
        
        self.log_file = os.path.join(log_dir, filename)
        self.filename = filename
        
        super().__init__(name, level)
    
    def _setup_handler(self) -> None:
        """Setup file handler for logging."""
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def set_log_dir(self, log_dir: str) -> None:
        """Update the log directory and move the log file."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create new log file path
        new_log_file = os.path.join(log_dir, self.filename)
        
        # Remove old handler
        if self.logger.handlers:
            old_handler = self.logger.handlers[0]
            old_handler.close()
            self.logger.removeHandler(old_handler)
        
        # Update paths
        self.log_dir = log_dir
        self.log_file = new_log_file
        
        # Setup new handler
        self._setup_handler()


class ConsoleLogger(BaseLogger):
    """
    Logger implementation that prints to console.
    """
    def __init__(self, name: str, level: str = "INFO"):
        super().__init__(name, level)
    
    def _setup_handler(self) -> None:
        """Setup console handler for logging."""
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)


class CompositeLogger(BaseLogger):
    """
    Logger that combines multiple loggers.
    Useful when you want to log to both file and console.
    """
    def __init__(self, name: str, 
                 log_dir: Optional[str] = None,
                 filename: Optional[str] = None,
                 level: str = "DEBUG"):  # Set default level to DEBUG for composite logger
        super().__init__(name, level)
        
        self.log_dir = log_dir
        self.filename = filename
        
        # Setup both handlers
        self._setup_console_handler()
        if log_dir is not None:
            self._setup_file_handler(log_dir, filename)
    
    def _setup_handler(self) -> None:
        """Not used in CompositeLogger as we set up handlers separately"""
        pass
    
    def _setup_console_handler(self) -> None:
        """Setup console handler"""
        handler = logging.StreamHandler()
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def _setup_file_handler(self, log_dir: str, filename: Optional[str] = None) -> None:
        """Setup file handler"""
        if not os.path.exists(log_dir):
            raise ValueError(f"Log directory {log_dir} does not exist")
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.name}_{timestamp}.log"
        self.filename = filename
        log_file = os.path.join(log_dir, filename)
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(self.formatter)
        self.logger.addHandler(handler)
    
    def set_log_dir(self, log_dir: str) -> None:
        """Update the log directory for the file handler."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Remove old file handler if it exists
        if len(self.logger.handlers) > 1:
            old_handler = self.logger.handlers[1]  # File handler is second
            old_handler.close()
            self.logger.removeHandler(old_handler)
        
        # Update log directory and setup new file handler
        self.log_dir = log_dir
        self._setup_file_handler(log_dir, self.filename)