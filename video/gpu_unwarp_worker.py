#!/usr/bin/env python3
"""
GPU Unwarp Worker Pipeline

Async worker that unwarps VR frames using GPU shaders in a separate thread.
Prevents blocking the main video processing pipeline.

Architecture:
    FFmpeg → input_queue → GPU Worker → output_queue → Tracker/YOLO

Usage:
    worker = GPUUnwarpWorker(projection_type='fisheye190')
    worker.start()

    # Main thread: Submit frames
    worker.submit_frame(frame_index, fisheye_frame)

    # Main thread: Get unwrapped frames
    frame_index, unwrapped_frame = worker.get_unwrapped_frame(timeout=0.1)

    # Cleanup
    worker.stop()
"""

import threading
import queue
import numpy as np
import cv2
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class FrameJob:
    """Frame unwarp job."""
    frame_index: int
    frame_data: np.ndarray
    timestamp_ms: float
    submit_time: float


class GPUUnwarpWorker:
    """
    Background worker that unwarps VR frames using GPU shaders.

    Runs in separate thread to avoid blocking main pipeline.
    """

    def __init__(self, projection_type: str = 'fisheye190',
                 output_size: int = 640,
                 queue_size: int = 16,
                 backend: str = 'auto',
                 pitch: float = 0.0,
                 yaw: float = 0.0,
                 roll: float = 0.0,
                 stereo_format: str = 'auto',
                 use_right_eye: bool = False,
                 batch_size: int = 4,
                 enable_double_buffering: bool = True,
                 input_format: str = 'bgr24'):
        """
        Initialize GPU unwarp worker.

        Args:
            projection_type: VR projection type ('fisheye190', 'fisheye200', 'equirect180')
            output_size: Output frame size (e.g., 640x640)
            queue_size: Max frames in queue (prevents memory overflow)
            backend: GPU backend ('metal', 'opengl', 'auto')
            pitch: Camera pitch angle in degrees (rotation around Y axis)
            yaw: Camera yaw angle in degrees (rotation around Z axis)
            roll: Camera roll angle in degrees (rotation around X axis)
            stereo_format: Stereo format ('auto', 'mono', 'sbs', 'tb')
            use_right_eye: Use right eye instead of left (for stereo)
            batch_size: Number of frames to pipeline (default 4 for 12% speedup)
            enable_double_buffering: Enable double buffering for textures
            input_format: Input pixel format ('bgr24' or 'rgba')
        """
        self.projection_type = projection_type
        self.output_size = output_size
        self.backend = backend
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll
        self.stereo_format = stereo_format
        self.use_right_eye = use_right_eye
        self.batch_size = max(1, min(8, batch_size))  # Clamp to 1-8
        self.enable_double_buffering = enable_double_buffering
        self.input_format = input_format

        # Queues
        self.input_queue = queue.Queue(maxsize=queue_size)
        self.output_queue = queue.Queue(maxsize=queue_size)

        # Worker thread
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # GPU context (created in worker thread)
        self.gpu_context = None

        # Reusable textures for input frames (double buffering)
        self.input_textures = []  # List of textures for double buffering
        self.output_textures = []  # Output texture pool (Metal backend)
        self.current_texture_index = 0
        self.current_output_index = 0
        self.last_input_size = None
        self.last_output_size = None

        # Batch processing
        self.frame_batch = []  # Accumulate frames for batch processing

        # Stats
        self.frames_processed = 0
        self.total_unwarp_time = 0.0
        self.avg_unwarp_time = 0.0

        self.logger = logging.getLogger(__name__)

    def start(self):
        """Start the GPU unwarp worker thread."""
        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.logger.warning("Worker thread already running")
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="GPUUnwarpWorker")
        self.worker_thread.start()
        self.logger.info(f"GPU unwarp worker started (projection={self.projection_type}, backend={self.backend}, "
                        f"batch_size={self.batch_size}, double_buffering={self.enable_double_buffering})")

    def stop(self, timeout: float = 2.0):
        """Stop the worker thread."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            return

        self.logger.info("Stopping GPU unwarp worker...")
        self.stop_event.set()
        self.worker_thread.join(timeout=timeout)

        if self.worker_thread.is_alive():
            self.logger.warning("Worker thread did not stop gracefully")

        self.logger.info(f"Worker stopped. Processed {self.frames_processed} frames, avg unwarp: {self.avg_unwarp_time*1000:.2f}ms")

    def submit_frame(self, frame_index: int, frame_data: np.ndarray, timestamp_ms: float = 0.0, timeout: float = 0.1) -> bool:
        """
        Submit a frame for GPU unwrapping.

        Args:
            frame_index: Frame number
            frame_data: Input fisheye/equirect frame (640x640 or original resolution)
            timestamp_ms: Frame timestamp
            timeout: Max time to wait if queue full

        Returns:
            True if submitted, False if queue full
        """
        job = FrameJob(
            frame_index=frame_index,
            frame_data=frame_data,
            timestamp_ms=timestamp_ms,
            submit_time=time.perf_counter()
        )

        try:
            self.input_queue.put(job, timeout=timeout)
            return True
        except queue.Full:
            self.logger.warning(f"Input queue full, dropping frame {frame_index}")
            return False

    def get_unwrapped_frame(self, timeout: float = 0.1) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        Get an unwrapped frame from the output queue.

        Args:
            timeout: Max time to wait for frame

        Returns:
            (frame_index, unwrapped_frame, timestamp_ms) or None if timeout
        """
        try:
            result = self.output_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return None

    def _worker_loop(self):
        """Main worker loop (runs in separate thread)."""
        try:
            # Initialize GPU context in worker thread
            self.logger.info("Initializing GPU context in worker thread...")
            self.gpu_context = self._init_gpu_context()

            if self.gpu_context is None:
                self.logger.error("Failed to initialize GPU context, worker stopping")
                return

            self.logger.info(f"GPU context initialized ({self.backend})")

            # Process frames
            while not self.stop_event.is_set():
                try:
                    # Get frame from input queue (with timeout to check stop_event)
                    job = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    # Process any remaining frames in batch on timeout
                    if self.frame_batch and self.batch_size > 1:
                        self._process_batch()
                    continue

                # Add to batch
                self.frame_batch.append(job)

                # Process batch when full (or batch_size=1 for immediate processing)
                if len(self.frame_batch) >= self.batch_size:
                    self._process_batch()

        except Exception as e:
            self.logger.error(f"Worker loop error: {e}", exc_info=True)

        finally:
            # Process any remaining frames in batch
            if self.frame_batch:
                self._process_batch()

            # Cleanup GPU context
            if self.gpu_context is not None:
                self._cleanup_gpu_context()

    def _process_batch(self):
        """Process accumulated frame batch."""
        if not self.frame_batch:
            return

        batch_start_time = time.perf_counter()

        # Process each frame in batch
        for job in self.frame_batch:
            # Unwarp using GPU
            start_time = time.perf_counter()
            unwrapped_frame = self._unwarp_frame_gpu(job.frame_data)
            unwarp_time = time.perf_counter() - start_time

            # Update stats
            self.frames_processed += 1
            self.total_unwarp_time += unwarp_time
            self.avg_unwarp_time = self.total_unwarp_time / self.frames_processed

            # Log slow frames
            if unwarp_time > 0.020:  # >20ms is slow for GPU unwarp
                self.logger.warning(f"Slow GPU unwarp: {unwarp_time*1000:.1f}ms for frame {job.frame_index}")

            # Submit to output queue (non-blocking drop if full)
            try:
                self.output_queue.put((job.frame_index, unwrapped_frame, job.timestamp_ms), block=False)
            except queue.Full:
                self.logger.warning(f"Output queue full, dropping frame {job.frame_index}")

        batch_time = time.perf_counter() - batch_start_time

        # Log batch performance
        if len(self.frame_batch) > 1 and self.frames_processed % 100 == 0:
            avg_frame_time = batch_time / len(self.frame_batch)
            self.logger.debug(f"Batch of {len(self.frame_batch)} frames: {batch_time*1000:.2f}ms total, {avg_frame_time*1000:.2f}ms/frame")

        # Clear batch
        self.frame_batch.clear()

    def _init_gpu_context(self):
        """
        Initialize GPU context for unwrapping.

        Returns GPU context object (Metal device, OpenGL context, etc.)
        """
        if self.backend == 'metal' or (self.backend == 'auto' and self._is_metal_available()):
            self.logger.info("Using Metal backend")
            return self._init_metal_context()

        elif self.backend == 'opengl' or (self.backend == 'auto' and self._is_opengl_available()):
            self.logger.info("Using OpenGL backend")
            return self._init_opengl_context()

        else:
            self.logger.error(f"No GPU backend available (tried: {self.backend})")
            return None

    def _init_metal_context(self):
        """Initialize Metal compute context (macOS)."""
        try:
            # Import Metal via PyObjC
            import objc
            from Metal import (
                MTLCreateSystemDefaultDevice,
                MTLResourceStorageModeShared,
                MTLPixelFormatRGBA8Unorm
            )

            device = MTLCreateSystemDefaultDevice()
            if device is None:
                self.logger.error("Failed to create Metal device")
                return None

            # Load compute shader
            shader_source = self._get_shader_source_metal()

            # Compile shader library
            library, error = device.newLibraryWithSource_options_error_(shader_source, None, None)
            if library is None:
                self.logger.error(f"Failed to compile Metal shader: {error}")
                return None

            # Create compute pipeline state
            kernel_function = library.newFunctionWithName_("unwarp_fisheye")
            if kernel_function is None:
                self.logger.error("Failed to find kernel function 'unwarp_fisheye'")
                return None

            pipeline_state, error = device.newComputePipelineStateWithFunction_error_(kernel_function, None)
            if pipeline_state is None:
                self.logger.error(f"Failed to create compute pipeline: {error}")
                return None

            # Create command queue
            command_queue = device.newCommandQueue()
            if command_queue is None:
                self.logger.error("Failed to create command queue")
                return None

            self.logger.info("Metal compute context initialized successfully")

            return {
                'backend': 'metal',
                'device': device,
                'pipeline_state': pipeline_state,
                'command_queue': command_queue,
                'library': library
            }

        except ImportError as e:
            self.logger.error(f"Metal not available (PyObjC not installed): {e}")
            return None
        except Exception as e:
            self.logger.error(f"Metal initialization failed: {e}", exc_info=True)
            return None

    def _init_opengl_context(self):
        """Initialize OpenGL context (cross-platform)."""
        try:
            import moderngl

            # Create headless OpenGL context
            ctx = moderngl.create_standalone_context(require=330)

            # Load shader
            vertex_shader, fragment_shader = self._get_shader_source_opengl()

            # Compile shader program
            program = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

            # Create framebuffer
            texture = ctx.texture((self.output_size, self.output_size), 3)  # RGB
            fbo = ctx.framebuffer(color_attachments=[texture])

            # Create fullscreen quad
            # Fullscreen quad in normalized device coordinates
            vertices = np.array([
                # Position (x, y)
                -1.0, -1.0,  # bottom-left
                 1.0, -1.0,  # bottom-right
                 1.0,  1.0,  # top-right
                -1.0,  1.0,  # top-left
            ], dtype='f4')

            indices = np.array([0, 1, 2, 0, 2, 3], dtype='i4')

            vbo = ctx.buffer(vertices.tobytes())
            ibo = ctx.buffer(indices.tobytes())

            # Create VAO with vertex attributes (only position)
            vao = ctx.vertex_array(
                program,
                [(vbo, '2f', 'in_position')],
                index_buffer=ibo
            )

            self.logger.info("OpenGL context initialized successfully")

            return {
                'backend': 'opengl',
                'ctx': ctx,
                'program': program,
                'fbo': fbo,
                'texture': texture,
                'vao': vao
            }

        except Exception as e:
            self.logger.error(f"OpenGL initialization failed: {e}", exc_info=True)
            return None

    def _unwarp_frame_gpu(self, fisheye_frame: np.ndarray) -> np.ndarray:
        """
        Unwarp frame using GPU shader.

        Args:
            fisheye_frame: Input fisheye/equirect frame

        Returns:
            Unwrapped rectilinear frame
        """
        if self.gpu_context is None:
            # Fallback: return input unchanged
            return fisheye_frame

        backend = self.gpu_context['backend']

        if backend == 'metal':
            return self._unwarp_metal(fisheye_frame)
        elif backend == 'opengl':
            return self._unwarp_opengl(fisheye_frame)
        else:
            return fisheye_frame

    def _unwarp_metal(self, frame: np.ndarray) -> np.ndarray:
        """Unwarp using Metal compute shader."""
        try:
            import objc
            from Metal import MTLPixelFormatRGBA8Unorm, MTLResourceStorageModeShared
            import struct

            device = self.gpu_context['device']
            pipeline_state = self.gpu_context['pipeline_state']
            command_queue = self.gpu_context['command_queue']

            frame_size = (frame.shape[1], frame.shape[0])

            # Initialize texture pool for double buffering
            if not self.input_textures or self.last_input_size != frame_size:
                # Release old textures
                self.input_textures.clear()

                # Create texture pool
                num_textures = 2 if self.enable_double_buffering else 1
                for _ in range(num_textures):
                    # Create input texture descriptor
                    tex_desc = objc.lookUpClass('MTLTextureDescriptor').alloc().init()
                    tex_desc.setTextureType_(2)  # MTLTextureType2D
                    tex_desc.setPixelFormat_(MTLPixelFormatRGBA8Unorm)
                    tex_desc.setWidth_(frame_size[0])
                    tex_desc.setHeight_(frame_size[1])
                    tex_desc.setUsage_(3)  # MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite

                    input_texture = device.newTextureWithDescriptor_(tex_desc)
                    self.input_textures.append(input_texture)

                self.last_input_size = frame_size
                self.current_texture_index = 0

            # Get current texture from pool
            input_texture = self.input_textures[self.current_texture_index]
            if self.enable_double_buffering:
                self.current_texture_index = (self.current_texture_index + 1) % len(self.input_textures)

            # Convert BGR to RGBA for Metal (skip if already RGBA)
            if self.input_format == 'rgba':
                # Input is already RGBA - use directly (no conversion overhead!)
                rgba_frame = frame if frame.shape[2] == 4 else np.dstack([frame, np.full(frame.shape[:2], 255, dtype=np.uint8)])
            else:
                # Convert BGR24 to RGBA
                rgba_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                rgba_frame[:, :, 0] = frame[:, :, 2]  # R
                rgba_frame[:, :, 1] = frame[:, :, 1]  # G
                rgba_frame[:, :, 2] = frame[:, :, 0]  # B
                rgba_frame[:, :, 3] = 255  # A

            # Upload frame data to texture
            from Metal import MTLRegionMake2D
            region = MTLRegionMake2D(0, 0, frame_size[0], frame_size[1])

            input_texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow_(
                region, 0, rgba_frame.tobytes(), frame_size[0] * 4
            )

            # Create output texture pool if needed (CRITICAL: reuse textures, don't create every frame!)
            if not self.output_textures or self.last_output_size != self.output_size:
                # Release old output textures
                self.output_textures.clear()

                # Create output texture pool
                num_out_textures = 2 if self.enable_double_buffering else 1
                for _ in range(num_out_textures):
                    tex_desc = objc.lookUpClass('MTLTextureDescriptor').alloc().init()
                    tex_desc.setTextureType_(2)
                    tex_desc.setPixelFormat_(MTLPixelFormatRGBA8Unorm)
                    tex_desc.setWidth_(self.output_size)
                    tex_desc.setHeight_(self.output_size)
                    tex_desc.setUsage_(3)
                    output_tex = device.newTextureWithDescriptor_(tex_desc)
                    self.output_textures.append(output_tex)

                self.last_output_size = self.output_size
                self.current_output_index = 0

            # Get current output texture from pool
            output_texture = self.output_textures[self.current_output_index]
            if self.enable_double_buffering:
                self.current_output_index = (self.current_output_index + 1) % len(self.output_textures)

            # Create uniforms buffer
            fov_degrees = float(self.projection_type.replace('fisheye', '')) if 'fisheye' in self.projection_type else 180.0
            # Apply FOV correction to match v360 filter behavior (shader interprets FOV differently)
            if 'fisheye' in self.projection_type:
                fov_degrees = fov_degrees / 1.5
            stereo_format_map = {'mono': 0, 'sbs': 1, 'tb': 2}
            h, w = frame.shape[:2]
            detected_format = self._detect_stereo_format(w, h)
            projection_type = 1 if 'fisheye' in self.projection_type else 0

            # Pack uniforms (must match Metal struct layout)
            # NOTE: Metal and OpenGL have same shader math, so pitch sign should match
            # Pitch correction: fisheye needs 0.5x, equirectangular needs 1.0x (no correction)
            pitch_corrected = self.pitch * 0.5 if 'fisheye' in self.projection_type else self.pitch

            uniforms_data = struct.pack(
                'fffiii',  # 6 values: 3 floats + 3 ints
                fov_degrees * np.pi / 180.0,  # fisheyeFOV
                90.0 * np.pi / 180.0,         # outputFOV
                float(pitch_corrected),        # pitch (corrected based on projection type)
                stereo_format_map.get(detected_format, 0),  # stereoFormat
                1 if self.use_right_eye else 0,  # useRightEye
                projection_type                   # projectionType
            )

            uniforms_buffer = device.newBufferWithBytes_length_options_(
                uniforms_data, len(uniforms_data), MTLResourceStorageModeShared
            )

            # Create command buffer and encoder
            command_buffer = command_queue.commandBuffer()
            compute_encoder = command_buffer.computeCommandEncoder()

            # Create sampler for bilinear filtering
            sampler_desc = objc.lookUpClass('MTLSamplerDescriptor').alloc().init()
            sampler_desc.setMinFilter_(1)  # MTLSamplerMinMagFilterLinear
            sampler_desc.setMagFilter_(1)  # MTLSamplerMinMagFilterLinear
            sampler_desc.setSAddressMode_(0)  # MTLSamplerAddressModeClampToEdge
            sampler_desc.setTAddressMode_(0)  # MTLSamplerAddressModeClampToEdge
            sampler = device.newSamplerStateWithDescriptor_(sampler_desc)

            # Set pipeline and resources
            compute_encoder.setComputePipelineState_(pipeline_state)
            compute_encoder.setTexture_atIndex_(input_texture, 0)
            compute_encoder.setTexture_atIndex_(output_texture, 1)
            compute_encoder.setSamplerState_atIndex_(sampler, 0)
            compute_encoder.setBuffer_offset_atIndex_(uniforms_buffer, 0, 0)

            # Dispatch compute kernel
            from Metal import MTLSizeMake
            thread_group_size = MTLSizeMake(16, 16, 1)
            thread_groups = MTLSizeMake(
                (self.output_size + 15) // 16,
                (self.output_size + 15) // 16,
                1
            )
            compute_encoder.dispatchThreadgroups_threadsPerThreadgroup_(thread_groups, thread_group_size)

            compute_encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            # Read back result
            output_data = bytearray(self.output_size * self.output_size * 4)
            region = MTLRegionMake2D(0, 0, self.output_size, self.output_size)

            output_texture.getBytes_bytesPerRow_fromRegion_mipmapLevel_(
                output_data, self.output_size * 4, region, 0
            )

            # Convert RGBA back to BGR
            rgba_output = np.frombuffer(output_data, dtype=np.uint8).reshape(
                (self.output_size, self.output_size, 4)
            )
            bgr_output = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
            bgr_output[:, :, 0] = rgba_output[:, :, 2]  # B
            bgr_output[:, :, 1] = rgba_output[:, :, 1]  # G
            bgr_output[:, :, 2] = rgba_output[:, :, 0]  # R

            return bgr_output

        except Exception as e:
            self.logger.error(f"Metal unwarp failed: {e}", exc_info=True)
            # Fallback to returning input frame
            return cv2.resize(frame, (self.output_size, self.output_size)) if frame.shape[0] != self.output_size else frame

    def _unwarp_opengl(self, frame: np.ndarray) -> np.ndarray:
        """Unwarp using OpenGL shader with double buffering."""
        import moderngl

        ctx = self.gpu_context['ctx']
        program = self.gpu_context['program']
        fbo = self.gpu_context['fbo']
        vao = self.gpu_context['vao']

        frame_size = (frame.shape[1], frame.shape[0])

        # Initialize texture pool for double buffering
        if not self.input_textures or self.last_input_size != frame_size:
            # Release old textures
            for tex in self.input_textures:
                tex.release()
            self.input_textures.clear()

            # Create texture pool (2 textures for double buffering, or more for batch processing)
            num_textures = 2 if self.enable_double_buffering else 1
            for _ in range(num_textures):
                tex = ctx.texture(frame_size, 3)
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
                tex.repeat_x = False
                tex.repeat_y = False
                self.input_textures.append(tex)

            self.last_input_size = frame_size
            self.current_texture_index = 0

        # Get current texture from pool (round-robin)
        current_texture = self.input_textures[self.current_texture_index]
        if self.enable_double_buffering:
            self.current_texture_index = (self.current_texture_index + 1) % len(self.input_textures)

        # Update texture data (much faster than creating new texture)
        # Flip Y for OpenGL coordinate system
        current_texture.write(np.flip(frame, axis=0).tobytes())

        # Get FOV and projection parameters from projection type
        if 'fisheye' in self.projection_type:
            fov_degrees = float(self.projection_type.replace('fisheye', ''))
            # Apply FOV correction to match v360 filter behavior (shader interprets FOV differently)
            fov_degrees = fov_degrees / 1.5
        elif 'equirect' in self.projection_type:
            fov_degrees = float(self.projection_type.replace('equirect', ''))
            if fov_degrees == 0:  # equirect180 or equirect360 without number
                fov_degrees = 180.0
        else:
            fov_degrees = 180.0  # default

        # Set uniforms
        current_texture.use(location=0)
        program['inputTexture'].value = 0
        program['fisheyeFOV'].value = fov_degrees * np.pi / 180.0
        program['outputFOV'].value = 90.0 * np.pi / 180.0  # Match v360 h_fov=90, v_fov=90

        # Pitch correction: fisheye needs 0.5x, equirectangular needs 1.0x (no correction)
        pitch_corrected = self.pitch * 0.5 if 'fisheye' in self.projection_type else self.pitch
        program['pitch'].value = float(-pitch_corrected)  # Negate for OpenGL coordinate system

        # Auto-detect stereo format from frame dimensions
        h, w = frame.shape[:2]
        detected_format = self._detect_stereo_format(w, h)

        # Stereo format: 0=mono, 1=SBS, 2=TB
        stereo_format_map = {'mono': 0, 'sbs': 1, 'tb': 2}
        program['stereoFormat'].value = stereo_format_map.get(detected_format, 0)
        program['useRightEye'].value = 1 if self.use_right_eye else 0

        # Projection type: 0=HE (half-equirect), 1=fisheye
        projection_type = 1 if 'fisheye' in self.projection_type else 0
        program['projectionType'].value = projection_type

        # Render to framebuffer
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0)
        vao.render(mode=moderngl.TRIANGLES)

        # Read back result (flip Y back)
        output_data = fbo.read(components=3)
        output_frame = np.frombuffer(output_data, dtype=np.uint8).reshape((self.output_size, self.output_size, 3))
        output_frame = np.flip(output_frame, axis=0).copy()

        # Don't release texture - we reuse it for next frame

        return output_frame

    def _detect_stereo_format(self, width: int, height: int) -> str:
        """
        Auto-detect stereo format from frame dimensions.

        Args:
            width: Frame width
            height: Frame height

        Returns:
            'mono', 'sbs', or 'tb'
        """
        # User override
        if self.stereo_format != 'auto':
            return self.stereo_format

        # Auto-detect based on aspect ratio
        aspect_ratio = width / height

        if aspect_ratio > 1.5:
            # Wide aspect ratio suggests SBS (e.g., 1280x640 = 2.0)
            return 'sbs'
        elif aspect_ratio < 0.67:
            # Tall aspect ratio suggests TB (e.g., 640x1280 = 0.5)
            return 'tb'
        else:
            # Square or close to square suggests mono (e.g., 640x640 = 1.0)
            return 'mono'

    def _get_shader_source_metal(self) -> str:
        """Get Metal compute shader source (ported from GLSL)."""
        return """
        #include <metal_stdlib>
        using namespace metal;

        #define PI 3.14159265359
        #define PI_2 1.57079632679

        // Uniforms structure
        struct Uniforms {
            float fisheyeFOV;
            float outputFOV;
            float pitch;
            int stereoFormat;  // 0=mono, 1=SBS, 2=TB
            int useRightEye;   // 0=left, 1=right
            int projectionType; // 0=HE, 1=fisheye
        };

        // Linear interpolation helper
        float lerp_helper(float y0, float y1, float x0, float x1, float x) {
            float m = (y1 - y0) / (x1 - x0);
            float b = y0;
            return m * (x - x0) + b;
        }

        // Rotation matrix around X-axis
        float3x3 rotationX(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return float3x3(
                1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s, c
            );
        }

        kernel void unwarp_fisheye(
            texture2d<float, access::sample> inputTexture [[texture(0)]],
            texture2d<float, access::write> outputTexture [[texture(1)]],
            sampler texSampler [[sampler(0)]],
            constant Uniforms &uniforms [[buffer(0)]],
            uint2 gid [[thread_position_in_grid]])
        {
            // Get output texture dimensions
            uint width = outputTexture.get_width();
            uint height = outputTexture.get_height();

            // Boundary check
            if (gid.x >= width || gid.y >= height) {
                return;
            }

            // Convert pixel coordinates to normalized device coordinates
            float x_flat_norm = lerp_helper(-1.0, 1.0, 0.0, float(width), float(gid.x));
            float y_flat_norm = lerp_helper(-1.0, 1.0, 0.0, float(height), float(gid.y));

            // Create 3D ray direction for flat perspective projection
            float focal_length = 1.0 / tan(uniforms.outputFOV * 0.5);
            float3 ray_dir = normalize(float3(x_flat_norm, -y_flat_norm, focal_length));

            // Apply pitch rotation
            float pitch_rad = uniforms.pitch * (PI / 180.0);
            float3x3 rotation = rotationX(pitch_rad);
            ray_dir = rotation * ray_dir;

            float2 pfish;

            if (uniforms.projectionType == 1) {
                // FISHEYE PROJECTION
                float p_x = ray_dir.x;
                float p_y = ray_dir.z;  // depth
                float p_z = ray_dir.y;  // height

                float p_xz = sqrt(p_x * p_x + p_z * p_z);
                float r = 2.0 * atan2(p_xz, p_y) / uniforms.fisheyeFOV;
                float theta = atan2(p_z, p_x);

                float x_src_norm = r * cos(theta);
                float y_src_norm = r * sin(theta);

                pfish.x = lerp_helper(0.0, 1.0, -1.0, 1.0, x_src_norm);
                pfish.y = lerp_helper(0.0, 1.0, -1.0, 1.0, y_src_norm);

                // Flip Y
                pfish.y = 1.0 - pfish.y;
            } else {
                // HALF-EQUIRECTANGULAR PROJECTION
                float longitude = atan2(ray_dir.x, ray_dir.z);
                float latitude = asin(clamp(ray_dir.y, -1.0, 1.0));

                pfish.x = 0.5 + (longitude / PI);
                pfish.y = 0.5 - (latitude / PI);
            }

            // Handle stereo formats
            if (uniforms.stereoFormat == 1) {
                // SBS
                if (uniforms.useRightEye == 1) {
                    pfish.x = pfish.x * 0.5 + 0.5;
                } else {
                    pfish.x = pfish.x * 0.5;
                }
            } else if (uniforms.stereoFormat == 2) {
                // TB
                if (uniforms.useRightEye == 1) {
                    pfish.y = pfish.y * 0.5 + 0.5;
                } else {
                    pfish.y = pfish.y * 0.5;
                }
            }

            // Clamp coordinates
            pfish = clamp(pfish, 0.0, 1.0);

            // Sample input texture with bilinear filtering
            float4 color = inputTexture.sample(texSampler, pfish);

            // Write to output
            outputTexture.write(color, gid);
        }
        """

    def _get_shader_source_opengl(self) -> Tuple[str, str]:
        """Get OpenGL shader source using proven flatten-fisheye-2.glsl approach."""
        vertex_shader = """
        #version 330

        in vec2 in_position;
        out vec2 v_texcoord;

        void main() {
            // Pass through position to fragment shader
            // Map from [-1, 1] to [0, 1] for texture coordinates
            v_texcoord = in_position * 0.5 + 0.5;
            gl_Position = vec4(in_position, 0.0, 1.0);
        }
        """

        fragment_shader = """
        #version 330
        #define PI 3.14159265359
        #define PI_2 1.57079632679

        uniform sampler2D inputTexture;
        uniform float fisheyeFOV;      // Field of view in radians for fisheye
        uniform float outputFOV;       // Output flat FOV in radians
        uniform float pitch;           // Pitch rotation in degrees
        uniform int stereoFormat;      // 0=mono, 1=SBS, 2=TB
        uniform int useRightEye;       // 0=left eye, 1=right eye
        uniform int projectionType;    // 0=HE (half-equirect), 1=fisheye

        in vec2 v_texcoord;
        out vec4 fragColor;

        float lerp(float y0, float y1, float x0, float x1, float x) {
            float m = (y1 - y0) / (x1 - x0);
            float b = y0;
            return m * (x - x0) + b;
        }

        mat3 rotationX(float angle) {
            float c = cos(angle);
            float s = sin(angle);
            return mat3(
                1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s, c
            );
        }

        void main() {
            // Based on flatten-fisheye-2.glsl (proven working shader)
            vec2 pfish;

            // Convert texture coordinates to normalized device coordinates
            float x_flat_norm = lerp(-1.0, 1.0, 0.0, 1.0, v_texcoord.x);
            float y_flat_norm = lerp(-1.0, 1.0, 0.0, 1.0, v_texcoord.y);

            // Create 3D ray direction for flat perspective projection
            float focal_length = 1.0 / tan(outputFOV * 0.5);
            vec3 ray_dir = normalize(vec3(x_flat_norm, -y_flat_norm, focal_length));

            // Apply pitch rotation (around X-axis)
            float pitch_rad = radians(pitch);
            mat3 rotation = rotationX(pitch_rad);
            ray_dir = rotation * ray_dir;

            if (projectionType == 1) {
                // FISHEYE PROJECTION
                // Convert 3D direction to fisheye coordinates
                float p_x = ray_dir.x;
                float p_y = ray_dir.z;  // depth
                float p_z = ray_dir.y;  // height

                // Calculate fisheye projection parameters
                float p_xz = sqrt(p_x * p_x + p_z * p_z);
                float r = 2.0 * atan(p_xz, p_y) / fisheyeFOV;
                float theta = atan(p_z, p_x);

                // Convert to normalized fisheye coordinates
                float x_src_norm = r * cos(theta);
                float y_src_norm = r * sin(theta);

                // Map to texture coordinates
                pfish.x = lerp(0.0, 1.0, -1.0, 1.0, x_src_norm);
                pfish.y = lerp(0.0, 1.0, -1.0, 1.0, y_src_norm);

                // Flip Y to correct upside-down output
                pfish.y = 1.0 - pfish.y;
            } else {
                // HALF-EQUIRECTANGULAR (HE) PROJECTION
                // Convert 3D ray to spherical coordinates
                float longitude = atan(ray_dir.x, ray_dir.z);  // azimuthal angle
                float latitude = asin(clamp(ray_dir.y, -1.0, 1.0));  // elevation angle

                // Map to texture coordinates [0, 1]
                // For 180° HE: width = π radians
                pfish.x = 0.5 + (longitude / PI);  // Map longitude to U
                pfish.y = 0.5 - (latitude / PI);   // Map latitude to V
            }

            // Handle stereo formats
            if (stereoFormat == 1) {
                // SBS (Side-by-Side): left eye in left half, right eye in right half
                if (useRightEye == 1) {
                    pfish.x = pfish.x * 0.5 + 0.5;  // Map to right half [0.5, 1.0]
                } else {
                    pfish.x = pfish.x * 0.5;        // Map to left half [0.0, 0.5]
                }
            } else if (stereoFormat == 2) {
                // TB (Top-Bottom): left eye on top, right eye on bottom
                if (useRightEye == 1) {
                    pfish.y = pfish.y * 0.5 + 0.5;  // Map to bottom half [0.5, 1.0]
                } else {
                    pfish.y = pfish.y * 0.5;        // Map to top half [0.0, 0.5]
                }
            }
            // stereoFormat == 0: mono, use full texture

            // Clamp to valid range
            pfish = clamp(pfish, 0.0, 1.0);

            fragColor = texture(inputTexture, pfish);
        }
        """

        return vertex_shader, fragment_shader

    def _cleanup_gpu_context(self):
        """Release GPU resources."""
        if self.gpu_context is None:
            return

        backend = self.gpu_context['backend']

        if backend == 'opengl':
            # Release input textures from pool
            for tex in self.input_textures:
                tex.release()
            self.input_textures.clear()

            # Release OpenGL resources
            ctx = self.gpu_context['ctx']
            ctx.release()

        elif backend == 'metal':
            # Release Metal textures
            for tex in self.input_textures:
                # Metal textures are ARC-managed, but clear the list
                pass
            self.input_textures.clear()

        self.logger.info("GPU context cleaned up")

    @staticmethod
    def _is_metal_available() -> bool:
        """Check if Metal is available (macOS)."""
        try:
            from Metal import MTLCreateSystemDefaultDevice
            device = MTLCreateSystemDefaultDevice()
            return device is not None
        except:
            return False

    @staticmethod
    def _is_opengl_available() -> bool:
        """Check if OpenGL is available."""
        try:
            import moderngl
            ctx = moderngl.create_standalone_context(require=330)
            ctx.release()
            return True
        except:
            return False


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Create worker
    worker = GPUUnwarpWorker(projection_type='fisheye190', output_size=640)
    worker.start()

    # Simulate frame processing
    for i in range(100):
        # Create dummy fisheye frame
        fisheye_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Submit to worker
        worker.submit_frame(i, fisheye_frame, timestamp_ms=i * 16.67)

        # Get unwrapped frame (non-blocking)
        result = worker.get_unwrapped_frame(timeout=0.001)
        if result:
            frame_idx, unwrapped, timestamp = result
            print(f"Got unwrapped frame {frame_idx}")

        time.sleep(0.001)  # Simulate processing time

    # Stop worker
    worker.stop()
