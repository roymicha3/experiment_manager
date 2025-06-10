"""Version utilities for database schema management."""
import re
from typing import Tuple
from dataclasses import dataclass

@dataclass
class SemanticVersion:
    """Represents a semantic version with major, minor, and patch components."""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __eq__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)
    
    def __lt__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __le__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) <= (other.major, other.minor, other.patch)
    
    def __gt__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) > (other.major, other.minor, other.patch)
    
    def __ge__(self, other: 'SemanticVersion') -> bool:
        return (self.major, self.minor, self.patch) >= (other.major, other.minor, other.patch)

class VersionError(Exception):
    """Error in version parsing or validation."""
    pass

def parse_version(version_str: str) -> SemanticVersion:
    """Parse a semantic version string into components.
    
    Args:
        version_str: Version string in format "major.minor.patch"
        
    Returns:
        SemanticVersion: Parsed version object
        
    Raises:
        VersionError: If version string is invalid
    """
    # Support both with and without 'v' prefix
    if version_str.startswith('v'):
        version_str = version_str[1:]
    
    pattern = r'^(\d+)\.(\d+)\.(\d+)$'
    match = re.match(pattern, version_str)
    
    if not match:
        raise VersionError(f"Invalid version format: {version_str}. Expected format: major.minor.patch")
    
    major, minor, patch = map(int, match.groups())
    return SemanticVersion(major, minor, patch)

def validate_version_string(version_str: str) -> bool:
    """Validate a version string format.
    
    Args:
        version_str: Version string to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        parse_version(version_str)
        return True
    except VersionError:
        return False

def is_compatible_upgrade(from_version: str, to_version: str) -> bool:
    """Check if upgrading from one version to another is compatible.
    
    Args:
        from_version: Current version string
        to_version: Target version string
        
    Returns:
        bool: True if upgrade is compatible (forward direction only)
        
    Raises:
        VersionError: If either version string is invalid
    """
    from_ver = parse_version(from_version)
    to_ver = parse_version(to_version)
    
    # Only allow forward upgrades
    return to_ver > from_ver

def is_backward_compatible(from_version: str, to_version: str) -> bool:
    """Check if a version change maintains backward compatibility.
    
    Args:
        from_version: Original version string
        to_version: New version string
        
    Returns:
        bool: True if backward compatible (only minor/patch increases)
        
    Raises:
        VersionError: If either version string is invalid
    """
    from_ver = parse_version(from_version)
    to_ver = parse_version(to_version)
    
    # Backward compatible if major version is same and minor/patch increased
    return (from_ver.major == to_ver.major and 
            to_ver >= from_ver)

def get_next_version(current_version: str, bump_type: str = "patch") -> str:
    """Generate the next version number based on bump type.
    
    Args:
        current_version: Current version string
        bump_type: Type of version bump ("major", "minor", "patch")
        
    Returns:
        str: Next version string
        
    Raises:
        VersionError: If current version is invalid or bump_type is unknown
    """
    version = parse_version(current_version)
    
    if bump_type == "major":
        return str(SemanticVersion(version.major + 1, 0, 0))
    elif bump_type == "minor":
        return str(SemanticVersion(version.major, version.minor + 1, 0))
    elif bump_type == "patch":
        return str(SemanticVersion(version.major, version.minor, version.patch + 1))
    else:
        raise VersionError(f"Unknown bump type: {bump_type}. Must be 'major', 'minor', or 'patch'")

def compare_versions(version1: str, version2: str) -> int:
    """Compare two version strings.
    
    Args:
        version1: First version string
        version2: Second version string
        
    Returns:
        int: -1 if version1 < version2, 0 if equal, 1 if version1 > version2
        
    Raises:
        VersionError: If either version string is invalid
    """
    ver1 = parse_version(version1)
    ver2 = parse_version(version2)
    
    if ver1 < ver2:
        return -1
    elif ver1 > ver2:
        return 1
    else:
        return 0 