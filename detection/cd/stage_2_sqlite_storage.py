import sqlite3
import json
import pickle
import logging
import os
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any
from contextlib import contextmanager, asynccontextmanager
import time

# Optional aiosqlite import
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
    aiosqlite = None

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
        """Initialize database schema optimized for performance (lean schema)."""
        with self.get_cursor() as cursor:
            # Lean frame objects table: only essentials required by Stage 3/Mixed
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS frame_objects (
                    frame_id INTEGER PRIMARY KEY,
                    atr_assigned_position TEXT,
                    atr_funscript_distance REAL,
                    locked_penis_x1 REAL,
                    locked_penis_y1 REAL,
                    locked_penis_x2 REAL,
                    locked_penis_y2 REAL,
                    locked_penis_active INTEGER,
                    contact_boxes_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) WITHOUT ROWID
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_frame_range
                ON frame_objects(frame_id)
                """
            )

            # ATR segments table unchanged
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS atr_segments (
                    id INTEGER PRIMARY KEY,
                    start_frame_id INTEGER,
                    end_frame_id INTEGER,
                    major_position TEXT,
                    confidence REAL,
                    duration INTEGER,
                    segment_data BLOB
                )
                """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_segment_range
                ON atr_segments(start_frame_id, end_frame_id)
                """
            )

            self._get_connection().commit()

    def store_frame_objects_batch(self, frame_objects: List[FrameObject], batch_size: int = 1000):
        """Store frame objects (lean schema)."""
        start_time = time.time()

        with self.get_cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION")

            batch_data = []
            for frame_obj in frame_objects:
                # Extract locked penis ROI if available
                lp_x1 = lp_y1 = lp_x2 = lp_y2 = None
                lp_active = 1 if getattr(frame_obj.atr_locked_penis_state, 'active', False) else 0
                try:
                    if getattr(frame_obj.atr_locked_penis_state, 'box', None):
                        bx = frame_obj.atr_locked_penis_state.box
                        lp_x1, lp_y1, lp_x2, lp_y2 = float(bx[0]), float(bx[1]), float(bx[2]), float(bx[3])
                except Exception:
                    lp_x1 = lp_y1 = lp_x2 = lp_y2 = None

                # Contact boxes as JSON (store essential fields directly)
                contact_json = None
                try:
                    contact_json = json.dumps(frame_obj.atr_detected_contact_boxes, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    contact_json = "[]"

                batch_data.append((
                    frame_obj.frame_id,
                    frame_obj.atr_assigned_position,
                    float(getattr(frame_obj, 'atr_funscript_distance', 0.0)),
                    lp_x1, lp_y1, lp_x2, lp_y2,
                    lp_active,
                    contact_json,
                ))

                if len(batch_data) >= batch_size:
                    cursor.executemany(
                        """
                        INSERT OR REPLACE INTO frame_objects
                        (frame_id, atr_assigned_position, atr_funscript_distance,
                         locked_penis_x1, locked_penis_y1, locked_penis_x2, locked_penis_y2,
                         locked_penis_active, contact_boxes_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        batch_data,
                    )
                    batch_data.clear()

            if batch_data:
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO frame_objects
                    (frame_id, atr_assigned_position, atr_funscript_distance,
                     locked_penis_x1, locked_penis_y1, locked_penis_x2, locked_penis_y2,
                     locked_penis_active, contact_boxes_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch_data,
                )

            cursor.execute("COMMIT")

        elapsed = time.time() - start_time
        self.logger.info(f"Stored {len(frame_objects)} frame objects in {elapsed:.2f}s (lean schema)")

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
            cursor.execute(
                """
                SELECT frame_id, atr_assigned_position, atr_funscript_distance,
                       locked_penis_x1, locked_penis_y1, locked_penis_x2, locked_penis_y2,
                       locked_penis_active, contact_boxes_json
                FROM frame_objects
                WHERE frame_id BETWEEN ? AND ?
                ORDER BY frame_id
                """,
                (start_frame, end_frame),
            )

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
            cursor.execute(
                """
                SELECT frame_id, atr_assigned_position, atr_funscript_distance,
                       locked_penis_x1, locked_penis_y1, locked_penis_x2, locked_penis_y2,
                       locked_penis_active, contact_boxes_json
                FROM frame_objects
                WHERE frame_id = ?
                """,
                (frame_id,),
            )

            row = cursor.fetchone()
            if row:
                return self._deserialize_frame_object(row)
        return None

    def _deserialize_frame_object(self, row) -> FrameObject:
        """Deserialize minimal frame object from lean schema row."""
        (
            frame_id,
            atr_assigned_position,
            atr_funscript_distance,
            lp_x1,
            lp_y1,
            lp_x2,
            lp_y2,
            lp_active,
            contact_boxes_json,
        ) = row

        # Reconstruct minimal FrameObject for Stage 3/Mixed consumption
        frame_obj = FrameObject(frame_id=frame_id, yolo_input_size=640)
        frame_obj.atr_assigned_position = atr_assigned_position
        frame_obj.atr_funscript_distance = atr_funscript_distance
        try:
            from detection.cd.stage_2_cd import ATRLockedPenisState
            frame_obj.atr_locked_penis_state = ATRLockedPenisState()
            if lp_active:
                frame_obj.atr_locked_penis_state.active = True
                if None not in (lp_x1, lp_y1, lp_x2, lp_y2):
                    frame_obj.atr_locked_penis_state.box = (lp_x1, lp_y1, lp_x2, lp_y2)
        except Exception:
            frame_obj.atr_locked_penis_state = None
        try:
            frame_obj.atr_detected_contact_boxes = json.loads(contact_boxes_json) if contact_boxes_json else []
        except Exception:
            frame_obj.atr_detected_contact_boxes = []
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

    def cleanup_database(self, remove_main_db: bool = False):
        """
        Clean up database files.
        
        Args:
            remove_main_db: If True, removes the main database file as well as temp files
        """
        # Always clean up temp files
        self.cleanup_temp_files()
        
        # Optionally remove main database
        if remove_main_db and self.db_path and os.path.exists(self.db_path):
            try:
                # Close connection first
                self.close()
                
                # Remove main database file
                os.remove(self.db_path)
                self.logger.info(f"Database file removed: {self.db_path}")
            except OSError as e:
                self.logger.warning(f"Failed to remove database file {self.db_path}: {e}")
            except Exception as e:
                self.logger.error(f"Error removing database: {e}")

    def __del__(self):
        self.close()


class AsyncStage2SQLiteStorage:
    """
    Async version of Stage2SQLiteStorage for high-performance non-blocking operations.
    Uses aiosqlite for async database operations with connection pooling.
    """

    def __init__(self, db_path: Optional[str], logger: Optional[logging.Logger] = None, max_connections: int = 5):
        if not AIOSQLITE_AVAILABLE:
            raise ImportError("aiosqlite is required for AsyncStage2SQLiteStorage. Install with: pip install aiosqlite")
        
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.max_connections = max_connections
        self._connection_pool = asyncio.Queue(maxsize=max_connections)
        self._pool_initialized = False
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="SQLite-Async")
        
    async def _init_pool(self):
        """Initialize the connection pool."""
        if self._pool_initialized or not self.db_path:
            return
            
        # Create database schema first with a temporary connection
        async with aiosqlite.connect(self.db_path) as conn:
            await self._setup_database_schema(conn)
        
        # Fill the connection pool
        for _ in range(self.max_connections):
            conn = await aiosqlite.connect(self.db_path)
            await self._optimize_connection(conn)
            await self._connection_pool.put(conn)
        
        self._pool_initialized = True
        self.logger.info(f"Async SQLite pool initialized with {self.max_connections} connections")

    async def _setup_database_schema(self, conn: aiosqlite.Connection):
        """Setup database schema."""
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS frame_objects (
                frame_id INTEGER PRIMARY KEY,
                data BLOB NOT NULL,
                serialization_method TEXT DEFAULT 'pickle',
                created_at REAL DEFAULT (julianday('now'))
            )
        ''')
        
        # Create indexes for performance
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_frame_id ON frame_objects(frame_id)')
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON frame_objects(created_at)')
        await conn.commit()

    async def _optimize_connection(self, conn: aiosqlite.Connection):
        """Apply performance optimizations to a connection."""
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=20000")
        await conn.execute("PRAGMA temp_store=MEMORY")
        await conn.execute("PRAGMA mmap_size=536870912")  # 512MB

    @asynccontextmanager
    async def _get_connection(self):
        """Get a connection from the pool."""
        if not self._pool_initialized:
            await self._init_pool()
        
        conn = await self._connection_pool.get()
        try:
            yield conn
        finally:
            await self._connection_pool.put(conn)

    async def store_frame_objects_batch_async(self, frame_objects: List[FrameObject], batch_size: int = 1000):
        """Store frame objects in batches asynchronously."""
        if not frame_objects:
            return

        start_time = time.time()
        total_objects = len(frame_objects)
        
        async with self._get_connection() as conn:
            # Process in batches to avoid memory issues
            for i in range(0, total_objects, batch_size):
                batch = frame_objects[i:i + batch_size]
                
                # Prepare batch data
                batch_data = []
                for frame_obj in batch:
                    try:
                        # Use pickle for serialization (faster than JSON for complex objects)
                        serialized_data = pickle.dumps(frame_obj)
                        batch_data.append((frame_obj.frame_id, serialized_data, 'pickle'))
                    except Exception as e:
                        self.logger.error(f"Failed to serialize frame {frame_obj.frame_id}: {e}")
                        continue
                
                # Insert batch
                await conn.executemany(
                    'INSERT OR REPLACE INTO frame_objects (frame_id, data, serialization_method) VALUES (?, ?, ?)',
                    batch_data
                )
                await conn.commit()
                
                self.logger.debug(f"Stored batch {i//batch_size + 1}/{(total_objects + batch_size - 1)//batch_size}")

        elapsed_time = time.time() - start_time
        self.logger.info(f"Async stored {total_objects} frame objects in {elapsed_time:.2f}s "
                        f"({total_objects/elapsed_time:.1f} objects/sec)")

    async def get_frame_objects_range_async(self, start_frame: int, end_frame: int) -> Dict[int, FrameObject]:
        """Get frame objects in a range asynchronously."""
        frame_objects = {}
        
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                'SELECT frame_id, data, serialization_method FROM frame_objects WHERE frame_id BETWEEN ? AND ? ORDER BY frame_id',
                (start_frame, end_frame)
            )
            
            async for row in cursor:
                frame_id, data, method = row
                try:
                    if method == 'pickle':
                        frame_obj = pickle.loads(data)
                    else:
                        # Fallback to JSON
                        frame_obj = FrameObject.from_dict(json.loads(data.decode('utf-8')))
                    frame_objects[frame_id] = frame_obj
                except Exception as e:
                    self.logger.error(f"Failed to deserialize frame {frame_id}: {e}")
                    continue
            
            await cursor.close()

        return frame_objects

    async def get_frame_count_async(self) -> int:
        """Get total number of stored frames asynchronously."""
        async with self._get_connection() as conn:
            cursor = await conn.execute('SELECT COUNT(*) FROM frame_objects')
            row = await cursor.fetchone()
            await cursor.close()
            return row[0] if row else 0

    async def cleanup_async(self):
        """Clean up async resources."""
        # Close all connections in the pool
        if self._pool_initialized:
            while not self._connection_pool.empty():
                try:
                    conn = self._connection_pool.get_nowait()
                    await conn.close()
                except asyncio.QueueEmpty:
                    break
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        self.logger.info("Async SQLite storage cleaned up")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            # Try to cleanup async resources
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup for later
                loop.create_task(self.cleanup_async())
            else:
                # Run cleanup directly
                loop.run_until_complete(self.cleanup_async())
        except Exception:
            pass  # Ignore errors during cleanup


class SQLiteConnectionPool:
    """
    Simple connection pool for Stage 3 workers to share SQLite connections efficiently.
    Reduces the overhead of creating individual connections per worker.
    """
    
    def __init__(self, db_path: str, max_connections: int = 10, logger: Optional[logging.Logger] = None):
        self.db_path = db_path
        self.max_connections = max_connections
        self.logger = logger or logging.getLogger(__name__)
        self._pool = []
        self._pool_lock = threading.RLock()
        self._total_connections = 0
        
        # Pre-create initial connections
        self._initialize_pool()
        
    def _initialize_pool(self):
        """Pre-create connections for the pool."""
        initial_size = min(3, self.max_connections)  # Start with 3 connections
        
        for _ in range(initial_size):
            try:
                conn = self._create_connection()
                self._pool.append(conn)
                self._total_connections += 1
            except Exception as e:
                self.logger.error(f"Failed to create initial pool connection: {e}")
                break
                
        self.logger.info(f"SQLite connection pool initialized with {len(self._pool)} connections")
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new optimized SQLite connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        
        # Apply optimizations
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=20000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        return conn
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (context manager)."""
        conn = None
        
        with self._pool_lock:
            if self._pool:
                # Reuse existing connection from pool
                conn = self._pool.pop()
            elif self._total_connections < self.max_connections:
                # Create new connection if under limit
                try:
                    conn = self._create_connection()
                    self._total_connections += 1
                    self.logger.debug(f"Created new pool connection (total: {self._total_connections})")
                except Exception as e:
                    self.logger.error(f"Failed to create new pool connection: {e}")
                    
        if conn is None:
            # Pool exhausted, wait and try again
            with self._pool_lock:
                if self._pool:
                    conn = self._pool.pop()
        
        if conn is None:
            # Still no connection, create temporary one
            self.logger.warning("Pool exhausted, creating temporary connection")
            conn = self._create_connection()
            temp_connection = True
        else:
            temp_connection = False
        
        try:
            yield conn
        finally:
            # Return connection to pool or close if temporary
            if temp_connection:
                conn.close()
            else:
                with self._pool_lock:
                    if len(self._pool) < self.max_connections:
                        self._pool.append(conn)
                    else:
                        # Pool is full, close this connection
                        conn.close()
                        self._total_connections -= 1
    
    def cleanup(self):
        """Close all connections in the pool."""
        with self._pool_lock:
            while self._pool:
                conn = self._pool.pop()
                try:
                    conn.close()
                except Exception:
                    pass
            self._total_connections = 0
        
        self.logger.info("SQLite connection pool cleaned up")


class PooledStage2SQLiteStorage:
    """
    Stage2SQLiteStorage that uses a shared connection pool for better resource management.
    Ideal for Stage 3 workers that need concurrent database access.
    """
    
    def __init__(self, connection_pool: SQLiteConnectionPool, logger: Optional[logging.Logger] = None):
        self.connection_pool = connection_pool
        self.logger = logger or logging.getLogger(__name__)
    
    def get_frame_objects_range(self, start_frame: int, end_frame: int) -> Dict[int, FrameObject]:
        """Get frame objects using pooled connection."""
        frame_objects = {}
        
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    'SELECT frame_id, data, serialization_method FROM frame_objects WHERE frame_id BETWEEN ? AND ? ORDER BY frame_id',
                    (start_frame, end_frame)
                )
                
                for row in cursor:
                    frame_id, data, method = row
                    try:
                        if method == 'pickle':
                            frame_obj = pickle.loads(data)
                        else:
                            # Fallback to JSON
                            frame_obj = FrameObject.from_dict(json.loads(data.decode('utf-8')))
                        frame_objects[frame_id] = frame_obj
                    except Exception as e:
                        self.logger.error(f"Failed to deserialize frame {frame_id}: {e}")
                        continue
            finally:
                cursor.close()
        
        return frame_objects
    
    def get_frame_count(self) -> int:
        """Get total number of stored frames using pooled connection."""
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT COUNT(*) FROM frame_objects')
                row = cursor.fetchone()
                return row[0] if row else 0
            finally:
                cursor.close()
