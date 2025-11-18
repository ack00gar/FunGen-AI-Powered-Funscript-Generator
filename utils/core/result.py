"""
Result type for operations that can fail.

Provides a clean alternative to exceptions for expected failures,
making error handling more explicit and easier to reason about.
"""

from typing import Generic, TypeVar, Optional, Callable
from dataclasses import dataclass


T = TypeVar('T')


@dataclass
class Result(Generic[T]):
    """
    Result type for operations that can fail.

    Similar to Rust's Result<T, E> or Swift's Result<Success, Failure>.

    Example:
        result = await fetch_scenes()
        if result.success:
            scenes = result.data
            process(scenes)
        else:
            show_error(result.error)
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

    @staticmethod
    def ok(data: T) -> 'Result[T]':
        """
        Create successful result.

        Args:
            data: Success data

        Returns:
            Result with success=True
        """
        return Result(success=True, data=data)

    @staticmethod
    def err(error: str) -> 'Result[T]':
        """
        Create error result.

        Args:
            error: Error message

        Returns:
            Result with success=False
        """
        return Result(success=False, error=error)

    def map(self, func: Callable[[T], 'U']) -> 'Result[U]':
        """
        Transform successful result using function.

        Args:
            func: Transform function

        Returns:
            Transformed result or error
        """
        if self.success:
            try:
                return Result.ok(func(self.data))
            except Exception as e:
                return Result.err(str(e))
        else:
            return Result(success=False, error=self.error)

    def or_else(self, default: T) -> T:
        """
        Get data or default value.

        Args:
            default: Default value if result is error

        Returns:
            Data if success, default if error
        """
        return self.data if self.success else default

    def unwrap(self) -> T:
        """
        Get data or raise exception.

        Returns:
            Data if success

        Raises:
            ValueError: If result is error
        """
        if self.success:
            return self.data
        else:
            raise ValueError(f"Result is error: {self.error}")
