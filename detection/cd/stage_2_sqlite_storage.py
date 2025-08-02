import sqlite3
import json
import pickle
import logging
import os
import threading
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import time

from detection.cd.stage_2_cd import FrameObject, ATRSegment


class Stage2SQLiteStorage:
    """
    High-performance SQLite storage for Stage 2 FrameObject data.
    Optimized for raw performance with minimal memory footprint.
    """

    def __init__(self, db_path: Optional[str], logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self._connection_cache = threading.local()
        if self.db_path:
            self._init_db()

    def set_db_path(self, db_path: str):
        """Set database path and initialize database."""
        self.db_path = db_path
        # Close any existing connections
        if hasattr(self._connection_cache, 'conn'):
            self._connection_cache.conn.close()
            delattr(self._connection_cache, 'conn')
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local connection with performance optimizations."""
        if not hasattr(self._connection_cache, 'conn'):
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )

            # Performance optimizations for raw speed
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=50000")  # Increased cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=1073741824")  # 1GB memory mapping
            conn.execute("PRAGMA page_size=65536")
            conn.execute("PRAGMA threads=4")  # Enable multi-threading
            conn.execute("PRAGMA optimize")  # Query optimization

            self._connection_cache.conn = conn

        return self._connection_cache.conn

    @contextmanager
    def get_cursor(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def _init_db(self):
        """Initialize database schema optimized for performance."""
        with self.get_cursor() as cursor:
            # Main frame objects table with minimal essential data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS frame_objects (
                    frame_id INTEGER PRIMARY KEY,
                    yolo_input_size INTEGER,
                    frame_width INTEGER,
                    frame_height INTEGER,
                    atr_assigned_position TEXT,

                    -- Essential ATR data (serialized as compact binary)
                    atr_locked_penis_state BLOB,
                    atr_detected_contact_boxes BLOB,

                    -- Boxes and poses (serialized as compact binary)
                    boxes_data BLOB,
                    poses_data BLOB,

                    -- Funscript related data
                    atr_funscript_distance REAL,
                    is_static_frame INTEGER DEFAULT 0,

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) WITHOUT ROWID
            """)

            # Index for range queries (critical for chunk processing)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_frame_range
                ON frame_objects(frame_id)
            """)

            # ATR segments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS atr_segments (
                    id INTEGER PRIMARY KEY,
                    start_frame_id INTEGER,
                    end_frame_id INTEGER,
                    major_position TEXT,
                    confidence REAL,
                    duration INTEGER,
                    segment_data BLOB
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_segment_range
                ON atr_segments(start_frame_id, end_frame_id)
            """)

            self._get_connection().commit()

    def store_frame_objects_batch(self, frame_objects: List[FrameObject], batch_size: int = 1000):
        """Store frame objects in optimized batches."""
        start_time = time.time()

        with self.get_cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION")

            batch_data = []
            for frame_obj in frame_objects:
                # Serialize complex objects to compact binary format
                atr_locked_penis_data = pickle.dumps(frame_obj.atr_locked_penis_state, protocol=pickle.HIGHEST_PROTOCOL)
                atr_contact_boxes_data = pickle.dumps(frame_obj.atr_detected_contact_boxes,
                                                      protocol=pickle.HIGHEST_PROTOCOL)
                boxes_data = pickle.dumps(frame_obj.boxes, protocol=pickle.HIGHEST_PROTOCOL)
                poses_data = pickle.dumps(frame_obj.poses, protocol=pickle.HIGHEST_PROTOCOL)

                batch_data.append((
                    frame_obj.frame_id,
                    frame_obj.yolo_input_size,
                    getattr(frame_obj, 'frame_width', 0),
                    getattr(frame_obj, 'frame_height', 0),
                    frame_obj.atr_assigned_position,
                    atr_locked_penis_data,
                    atr_contact_boxes_data,
                    boxes_data,
                    poses_data,
                    frame_obj.atr_funscript_distance,
                    1 if getattr(frame_obj, 'is_static_frame', False) else 0
                ))

                if len(batch_data) >= batch_size:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO frame_objects
                        (frame_id, yolo_input_size, frame_width, frame_height,
                         atr_assigned_position, atr_locked_penis_state, atr_detected_contact_boxes,
                         boxes_data, poses_data, atr_funscript_distance, is_static_frame)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, batch_data)
                    batch_data.clear()

            # Insert remaining batch
            if batch_data:
                cursor.executemany("""
                    INSERT OR REPLACE INTO frame_objects
                    (frame_id, yolo_input_size, frame_width, frame_height,
                     atr_assigned_position, atr_locked_penis_state, atr_detected_contact_boxes,
                     boxes_data, poses_data, atr_funscript_distance, is_static_frame)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)

            cursor.execute("COMMIT")

        elapsed = time.time() - start_time
        self.logger.info(f"Stored {len(frame_objects)} frame objects in {elapsed:.2f}s")

    def store_atr_segments(self, segments: List[ATRSegment]):
        """Store ATR segments."""
        with self.get_cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION")

            for segment in segments:
                segment_data = pickle.dumps(segment, protocol=pickle.HIGHEST_PROTOCOL)
                cursor.execute("""
                    INSERT OR REPLACE INTO atr_segments
                    (id, start_frame_id, end_frame_id, major_position, confidence, duration, segment_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    segment.id,
                    segment.start_frame_id,
                    segment.end_frame_id,
                    segment.major_position,
                    getattr(segment, 'confidence', 0.0),
                    segment.duration,
                    segment_data
                ))

            cursor.execute("COMMIT")

        self.logger.info(f"Stored {len(segments)} ATR segments")

    def get_frame_objects_range(self, start_frame: int, end_frame: int) -> Dict[int, FrameObject]:
        """Get frame objects in range with optimized query and connection reuse."""
        start_time = time.time()

        # Use optimized query with prepared statement pattern
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Single optimized query with minimal data transfer
            cursor.execute("""
                SELECT frame_id, yolo_input_size, frame_width, frame_height,
                       atr_assigned_position, atr_locked_penis_state, atr_detected_contact_boxes,
                       boxes_data, poses_data, atr_funscript_distance, is_static_frame
                FROM frame_objects
                WHERE frame_id BETWEEN ? AND ?
                ORDER BY frame_id
            """, (start_frame, end_frame))

            # Use fetchmany for better memory usage on large ranges
            frame_objects = {}
            while True:
                rows = cursor.fetchmany(1000)  # Process in batches
                if not rows:
                    break

                for row in rows:
                    frame_obj = self._deserialize_frame_object(row)
                    frame_objects[frame_obj.frame_id] = frame_obj

        finally:
            cursor.close()

        elapsed = time.time() - start_time
        if elapsed > 0.1:  # Only log slower queries
            self.logger.debug(
                f"Loaded {len(frame_objects)} frame objects ({start_frame}-{end_frame}) in {elapsed:.3f}s")

        return frame_objects

    def get_frame_object(self, frame_id: int) -> Optional[FrameObject]:
        """Get single frame object by ID."""
        with self.get_cursor() as cursor:
            cursor.execute("""
                SELECT frame_id, yolo_input_size, frame_width, frame_height,
                       atr_assigned_position, atr_locked_penis_state, atr_detected_contact_boxes,
                       boxes_data, poses_data, atr_funscript_distance, is_static_frame
                FROM frame_objects
                WHERE frame_id = ?
            """, (frame_id,))

            row = cursor.fetchone()
            if row:
                return self._deserialize_frame_object(row)
        return None

    def _deserialize_frame_object(self, row) -> FrameObject:
        """Deserialize frame object from database row with optimized unpickling."""
        (frame_id, yolo_input_size, frame_width, frame_height,
         atr_assigned_position, atr_locked_penis_data, atr_contact_boxes_data,
         boxes_data, poses_data, atr_funscript_distance, is_static_frame) = row

        # Create frame object
        frame_obj = FrameObject(frame_id=frame_id, yolo_input_size=yolo_input_size)
        frame_obj.frame_width = frame_width
        frame_obj.frame_height = frame_height
        frame_obj.atr_assigned_position = atr_assigned_position
        frame_obj.atr_funscript_distance = atr_funscript_distance
        frame_obj.is_static_frame = bool(is_static_frame)

        # Deserialize complex objects with fast unpickling
        try:
            frame_obj.atr_locked_penis_state = pickle.loads(atr_locked_penis_data)
            frame_obj.atr_detected_contact_boxes = pickle.loads(atr_contact_boxes_data)
            frame_obj.boxes = pickle.loads(boxes_data)
            frame_obj.poses = pickle.loads(poses_data)
        except Exception as e:
            self.logger.warning(f"Failed to deserialize frame {frame_id}: {e}")
            # Fallback to empty objects
            frame_obj.atr_locked_penis_state = None
            frame_obj.atr_detected_contact_boxes = []
            frame_obj.boxes = []
            frame_obj.poses = []

        return frame_obj

    def get_atr_segments(self) -> List[ATRSegment]:
        """Get all ATR segments."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT segment_data FROM atr_segments ORDER BY start_frame_id")
            segments = []
            for (segment_data,) in cursor.fetchall():
                segments.append(pickle.loads(segment_data))
            return segments

    def get_frame_count(self) -> int:
        """Get total number of stored frame objects."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM frame_objects")
            return cursor.fetchone()[0]

    def get_frame_range(self) -> tuple:
        """Get min and max frame IDs."""
        with self.get_cursor() as cursor:
            cursor.execute("SELECT MIN(frame_id), MAX(frame_id) FROM frame_objects")
            result = cursor.fetchone()
            return result if result[0] is not None else (0, 0)

    def optimize_database(self):
        """Run database optimization."""
        with self.get_cursor() as cursor:
            cursor.execute("ANALYZE")
            cursor.execute("VACUUM")
        self.logger.info("Database optimized")

    def close(self):
        """Close all connections."""
        if hasattr(self._connection_cache, 'conn'):
            self._connection_cache.conn.close()

    def get_frame_objects_streaming(self, start_frame: int, end_frame: int, batch_size: int = 500):
        """Generator that yields frame objects in batches for memory-efficient processing."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT frame_id, yolo_input_size, frame_width, frame_height,
                       atr_assigned_position, atr_locked_penis_state, atr_detected_contact_boxes,
                       boxes_data, poses_data, atr_funscript_distance, is_static_frame
                FROM frame_objects
                WHERE frame_id BETWEEN ? AND ?
                ORDER BY frame_id
            """, (start_frame, end_frame))

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break

                batch = {}
                for row in rows:
                    frame_obj = self._deserialize_frame_object(row)
                    batch[frame_obj.frame_id] = frame_obj

                yield batch

        finally:
            cursor.close()

    def cleanup_temp_files(self):
        """Clean up temporary database files."""
        temp_files = [f"{self.db_path}-wal", f"{self.db_path}-shm"]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def __del__(self):
        self.close()
