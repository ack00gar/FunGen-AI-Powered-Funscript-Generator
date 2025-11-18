#!/usr/bin/env python3
"""
Async Video Processor (Progressive Improvement - Stage 3)

Demonstrates migration from threading to async/await for:
- Better CPU utilization (no GIL contention)
- Cleaner code (coroutines vs threads)
- Easier testing and debugging
- Non-blocking I/O operations

This is a reference implementation showing the architectural pattern.
Full integration would require more extensive refactoring.
"""

import asyncio
import subprocess
import numpy as np
import cv2
from typing import Optional, AsyncIterator
from dataclasses import dataclass
import logging


@dataclass
class FrameData:
    """Frame data container."""
    index: int
    data: np.ndarray
    timestamp_ms: float


class AsyncFFmpegReader:
    """
    Async FFmpeg frame reader using asyncio subprocess.

    Advantages over threading:
    - Non-blocking I/O without GIL contention
    - Better resource utilization
    - Easier cancellation and cleanup
    """

    def __init__(self, video_path: str, output_size: int = 640):
        """
        Initialize async FFmpeg reader.

        Args:
            video_path: Path to video file
            output_size: Output frame size (square)
        """
        self.video_path = video_path
        self.output_size = output_size
        self.frame_size_bytes = output_size * output_size * 3

        self.process: Optional[asyncio.subprocess.Process] = None
        self.current_frame_index = 0
        self.fps = 30.0

        self.logger = logging.getLogger(__name__)

    async def start(self, start_frame: int = 0) -> bool:
        """
        Start FFmpeg process asynchronously.

        Args:
            start_frame: Frame to start from

        Returns:
            True if started successfully
        """
        try:
            # Build FFmpeg command
            start_time = start_frame / self.fps if start_frame > 0 else 0

            cmd = [
                'ffmpeg',
                '-loglevel', 'warning',
                '-hwaccel', 'auto'
            ]

            if start_time > 0:
                cmd.extend(['-ss', f'{start_time:.3f}'])

            cmd.extend([
                '-i', self.video_path,
                '-vf', f'scale={self.output_size}:{self.output_size}',
                '-pix_fmt', 'bgr24',
                '-f', 'rawvideo',
                'pipe:1'
            ])

            # Start async subprocess
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            self.current_frame_index = start_frame
            self.logger.info(f"Started async FFmpeg reader at frame {start_frame}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
            return False

    async def read_frame(self) -> Optional[FrameData]:
        """
        Read next frame asynchronously.

        Returns:
            FrameData or None if end of stream
        """
        if self.process is None or self.process.stdout is None:
            return None

        try:
            # Non-blocking read with timeout
            frame_bytes = await asyncio.wait_for(
                self.process.stdout.readexactly(self.frame_size_bytes),
                timeout=5.0
            )

            # Convert to numpy array
            frame = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = frame.reshape((self.output_size, self.output_size, 3))

            # Create frame data
            frame_data = FrameData(
                index=self.current_frame_index,
                data=frame,
                timestamp_ms=self.current_frame_index / self.fps * 1000
            )

            self.current_frame_index += 1

            return frame_data

        except asyncio.TimeoutError:
            self.logger.warning("Frame read timeout")
            return None

        except asyncio.IncompleteReadError:
            # End of stream
            return None

        except Exception as e:
            self.logger.error(f"Error reading frame: {e}")
            return None

    async def read_frames(self) -> AsyncIterator[FrameData]:
        """
        Async generator for reading frames.

        Yields:
            FrameData for each frame
        """
        while True:
            frame_data = await self.read_frame()
            if frame_data is None:
                break
            yield frame_data

    async def seek(self, frame_index: int) -> bool:
        """
        Seek to specific frame.

        Args:
            frame_index: Target frame index

        Returns:
            True if seek successful
        """
        # Stop current stream
        await self.stop()

        # Start new stream at target frame
        return await self.start(frame_index)

    async def stop(self) -> None:
        """Stop FFmpeg process gracefully."""
        if self.process is None:
            return

        try:
            # Terminate process
            self.process.terminate()

            # Wait for termination with timeout
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                # Force kill if not terminated
                self.process.kill()
                await self.process.wait()

            self.logger.info("Stopped async FFmpeg reader")

        except Exception as e:
            self.logger.error(f"Error stopping FFmpeg: {e}")

        finally:
            self.process = None


class AsyncVideoProcessor:
    """
    Async video processor with frame buffering and caching.

    Demonstrates async/await patterns for video processing.
    """

    def __init__(self, video_path: str, output_size: int = 640):
        """
        Initialize async video processor.

        Args:
            video_path: Path to video file
            output_size: Output frame size
        """
        self.video_path = video_path
        self.output_size = output_size

        self.reader = AsyncFFmpegReader(video_path, output_size)

        # Async frame buffer
        self.frame_buffer = asyncio.Queue(maxsize=10)

        # Background tasks
        self.buffer_task: Optional[asyncio.Task] = None

        self.logger = logging.getLogger(__name__)

    async def start(self) -> bool:
        """
        Start video processor.

        Returns:
            True if started successfully
        """
        # Start FFmpeg reader
        if not await self.reader.start():
            return False

        # Start background buffer task
        self.buffer_task = asyncio.create_task(self._buffer_frames())

        return True

    async def _buffer_frames(self) -> None:
        """
        Background task to buffer frames from FFmpeg.

        This runs concurrently and fills the frame buffer.
        """
        try:
            async for frame_data in self.reader.read_frames():
                # Add to buffer (blocks if buffer full)
                await self.frame_buffer.put(frame_data)

        except asyncio.CancelledError:
            self.logger.info("Frame buffering cancelled")

        except Exception as e:
            self.logger.error(f"Error in frame buffering: {e}")

    async def get_next_frame(self) -> Optional[FrameData]:
        """
        Get next frame from buffer.

        Returns:
            FrameData or None if no more frames
        """
        try:
            # Get from buffer with timeout
            frame_data = await asyncio.wait_for(
                self.frame_buffer.get(),
                timeout=2.0
            )
            return frame_data

        except asyncio.TimeoutError:
            return None

    async def seek_to_frame(self, frame_index: int) -> Optional[FrameData]:
        """
        Seek to specific frame.

        Args:
            frame_index: Target frame index

        Returns:
            First frame after seek or None
        """
        # Cancel buffer task
        if self.buffer_task:
            self.buffer_task.cancel()
            try:
                await self.buffer_task
            except asyncio.CancelledError:
                pass

        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Seek reader
        if not await self.reader.seek(frame_index):
            return None

        # Restart buffer task
        self.buffer_task = asyncio.create_task(self._buffer_frames())

        # Get first frame
        return await self.get_next_frame()

    async def stop(self) -> None:
        """Stop video processor."""
        # Cancel buffer task
        if self.buffer_task:
            self.buffer_task.cancel()
            try:
                await self.buffer_task
            except asyncio.CancelledError:
                pass

        # Stop reader
        await self.reader.stop()


# Example usage
async def main():
    """Example usage of async video processor."""
    print("Async Video Processor Example\n" + "="*60 + "\n")

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create processor
    processor = AsyncVideoProcessor('/path/to/video.mp4', output_size=640)

    # Start processor
    print("Starting video processor...")
    if not await processor.start():
        print("Failed to start processor")
        return

    # Read 100 frames
    print("Reading 100 frames...\n")
    frame_count = 0

    while frame_count < 100:
        frame_data = await processor.get_next_frame()
        if frame_data is None:
            break

        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Frame {frame_data.index}: {frame_data.data.shape} @ {frame_data.timestamp_ms:.2f}ms")

    # Seek to frame 500
    print(f"\nSeeking to frame 500...")
    frame_data = await processor.seek_to_frame(500)
    if frame_data:
        print(f"Seeked to frame {frame_data.index}")

    # Read 10 more frames
    print("\nReading 10 more frames after seek...\n")
    for _ in range(10):
        frame_data = await processor.get_next_frame()
        if frame_data:
            print(f"Frame {frame_data.index}: {frame_data.data.shape}")

    # Stop processor
    print("\nStopping processor...")
    await processor.stop()

    print("\nDone!")


if __name__ == '__main__':
    asyncio.run(main())
