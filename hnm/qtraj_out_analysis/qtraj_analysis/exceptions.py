"""
Custom exception hierarchy for qtraj_analysis.
Provides physics-aware error messages and debugging context.

Usage:
    from qtraj_analysis.exceptions import DataFileParsingError
    
    raise DataFileParsingError(
        "Invalid data format",
        context={"line_number": 42, "expected": "float", "got": "string"}
    )
"""

class QTrajError(Exception):
    """
    Base exception for all qtraj_analysis errors.
    
    Attributes:
        message: Human-readable error description
        context: Dictionary with debugging information
    """
    def __init__(self, message: str, context: dict = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self):
        """Format error with context information."""
        base_msg = self.message
        if self.context:
            context_lines = [f"  {k}: {v}" for k, v in self.context.items()]
            context_str = "\n".join(context_lines)
            return f"{base_msg}\n\nContext:\n{context_str}"
        return base_msg
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.message!r}, context={self.context!r})"


# ============================================================================
# I/O and Data File Errors
# ============================================================================

class DataFileError(QTrajError):
    """Base class for data file related errors."""
    pass


class DataFileNotFoundError(DataFileError):
    """Data file doesn't exist at specified path."""
    pass


class DataFileFormatError(DataFileError):
    """Data file has incorrect format or structure."""
    pass


class DataFileParsingError(DataFileError):
    """Error parsing data file content (e.g., non-numeric values)."""
    pass


class GlauberFileError(DataFileError):
    """Glauber interpolation table file error."""
    pass


# ============================================================================
# Physics Calculation Errors
# ============================================================================

class PhysicsError(QTrajError):
    """Base class for physics calculation errors."""
    pass


class TrajectoryMatchingError(PhysicsError):
    """
    S-wave/P-wave trajectory matching failed.
    
    Common causes:
    - L column not in expected position
    - Missing S or P records for some trajectories
    - Meta-data keying issues (floating-point precision)
    """
    pass


class GlauberModelError(PhysicsError):
    """
    Glauber model interpolation error.
    
    Common causes:
    - Interpolation tables have incorrect format
    - b-values outside interpolation range (extrapolation)
    - Inconsistent b vs c mapping
    """
    pass


class FeedDownError(PhysicsError):
    """
    Feed-down matrix computation error.
    
    Common causes:
    - Singular feed-down matrix (cannot invert)
    - Inconsistent state ordering
    - Invalid branching ratios
    """
    pass


class StatisticalError(PhysicsError):
    """
    Statistical computation error.
    
    Common causes:
    - Zero trajectories in bin
    - All weights are zero
    - Insufficient statistics for uncertainty estimation
    """
    pass


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(QTrajError):
    """Base class for input validation failures."""
    pass


class ArrayShapeError(ValidationError):
    """
    Array has incorrect shape.
    
    Attributes:
        array_name: Name of the problematic array
        expected_shape: Expected shape (tuple or description)
        actual_shape: Actual shape from the array
    """
    def __init__(self, array_name: str, expected_shape, actual_shape):
        message = f"Array '{array_name}' has incorrect shape"
        context = {
            "array_name": array_name,
            "expected_shape": str(expected_shape),
            "actual_shape": str(actual_shape),
        }
        super().__init__(message, context)
        self.array_name = array_name
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class ArrayValueError(ValidationError):
    """
    Array contains invalid values.
    
    Common cases:
    - NaN or Inf values
    - Negative values where only positive expected
    - Values outside physical range (e.g., survival > 1.0)
    """
    pass


class PhysicsConstraintError(ValidationError):
    """
    Input violates physics constraint.
    
    Examples:
    - Impact parameter b < 0 or b > 30 fm
    - Survival probability outside [0, 1]
    - Negative cross-section
    """
    pass


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(QTrajError):
    """
    Configuration is invalid or incomplete.
    
    Common causes:
    - Missing required parameters
    - Inconsistent array lengths (e.g., npart_vals vs bvals)
    - Invalid parameter combinations
    """
    pass


class MissingDependencyError(QTrajError):
    """
    Required Python dependency not installed.
    
    Example:
        raise MissingDependencyError(
            "scipy is required for Glauber interpolation",
            context={"package": "scipy", "install_cmd": "pip install scipy"}
        )
    """
    pass


# ============================================================================
# Utility Functions
# ============================================================================

def add_file_context(exception: QTrajError, filepath: str, line_number: int = None) -> QTrajError:
    """
    Add file context to an existing exception.
    
    Args:
        exception: Exception to augment
        filepath: Path to file being processed
        line_number: Optional line number in file
    
    Returns:
        Modified exception with added context
    
    Example:
        try:
            value = float(token)
        except ValueError as e:
            raise add_file_context(
                DataFileParsingError("Non-numeric value"),
                filepath="data.txt",
                line_number=42
            ) from e
    """
    exception.context["file"] = filepath
    if line_number is not None:
        exception.context["line_number"] = line_number
    return exception


def wrap_exception(
    original: Exception,
    new_exception_class: type,
    message: str,
    context: dict = None
) -> QTrajError:
    """
    Wrap a generic exception in a domain-specific exception.
    
    Args:
        original: Original exception that was caught
        new_exception_class: QTrajError subclass to raise
        message: New error message
        context: Additional context dictionary
    
    Returns:
        New exception with original as __cause__
    
    Example:
        try:
            data = np.loadtxt(path)
        except Exception as e:
            raise wrap_exception(
                e,
                DataFileError,
                "Failed to load numeric data",
                context={"path": path}
            ) from e
    """
    ctx = context or {}
    ctx["original_error"] = str(original)
    ctx["original_type"] = type(original).__name__
    return new_exception_class(message, ctx)
