import imgui
import logging
from typing import Optional, Tuple

import config.constants as constants
from config.element_group_colors import VideoDisplayColors
from application.utils import get_logo_texture_manager, get_icon_texture_manager
from application.utils.imgui_helpers import DisabledScope as _DisabledScope
from application.utils.feature_detection import is_feature_available as _is_feature_available

# Module-level logger for Handy debug output (disabled by default)
_handy_debug_logger = logging.getLogger(__name__ + '.handy')



class HandyVideoMixin:
    """Mixin fragment for VideoDisplayUI."""

    def _is_handy_available(self):
        """Check if Handy devices are connected and device control is enabled."""
        try:
            # Check if device control is enabled
            if not hasattr(self.app, 'app_settings'):
                return False
                
            device_control_enabled = self.app.app_settings.get("device_control_video_playback", False)
            if not device_control_enabled:
                return False
                
            # Check if device manager exists and has Handy devices
            if not hasattr(self.app, 'device_manager') or not self.app.device_manager:
                return False
                
            device_manager = self.app.device_manager
            if not device_manager.is_connected():
                return False
                
            # Check for connected Handy devices
            for device_id, backend in device_manager.connected_devices.items():
                device_info = backend.get_device_info()
                if device_info and "handy" in device_info.name.lower():
                    return True
                    
            return False
            
        except Exception:
            return False
    

    def _has_funscript_actions(self):
        """Check if funscript actions are available for streaming."""
        try:
            # Check if funscript processor exists
            if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
                return False
                
            fs_proc = self.app.funscript_processor
            
            # Get the MultiAxisFunscript object
            funscript_obj = fs_proc.get_funscript_obj()
            if not funscript_obj:
                return False
                
            # Check primary axis actions only (Handy uses primary axis)
            primary_actions = fs_proc.get_actions('primary')
            return len(primary_actions) > 0 if primary_actions else False
            
        except Exception as e:
            return False
    

    def _start_handy_streaming(self):
        """Start Handy streaming with current funscript and video position."""
        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" _start_handy_streaming() called")
        
        # Force video to real-time speed for proper Handy synchronization
        if hasattr(self.app, 'app_state_ui') and hasattr(self.app.app_state_ui, 'selected_processing_speed_mode'):
            # Save current speed mode to restore later
            self.saved_processing_speed_mode = self.app.app_state_ui.selected_processing_speed_mode
            # Force to real-time speed
            self.app.app_state_ui.selected_processing_speed_mode = constants.ProcessingSpeedMode.REALTIME
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Forced video speed to REALTIME (was {self.saved_processing_speed_mode.value})")
        
        import threading
        import asyncio
        
        def start_streaming_async():
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" start_streaming_async() thread started")
            loop = None
            try:
                # Set preparing state
                self.handy_preparing = True
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" handy_preparing set to True")
                
                # Get current video position with multiple fallback methods
                current_time_ms = 0.0
                current_frame = 0
                fps = 0.0
                
                if hasattr(self.app, 'processor') and self.app.processor:
                    # Method 1: Direct frame index and FPS
                    if hasattr(self.app.processor, 'current_frame_index'):
                        current_frame = self.app.processor.current_frame_index
                        
                    if hasattr(self.app.processor, 'fps'):
                        fps = self.app.processor.fps
                        
                    # Method 2: Try video_info if available
                    if hasattr(self.app.processor, 'video_info') and self.app.processor.video_info:
                        if fps <= 0 and 'fps' in self.app.processor.video_info:
                            fps = self.app.processor.video_info['fps']
                            
                    # Method 3: Try get_current_frame_timestamp_ms if available
                    if hasattr(self.app.processor, 'get_current_frame_timestamp_ms'):
                        try:
                            timestamp_ms = self.app.processor.get_current_frame_timestamp_ms()
                            if timestamp_ms > 0:
                                current_time_ms = timestamp_ms
                        except Exception:
                            pass
                    
                    # Calculate from frame and FPS if timestamp method didn't work
                    if current_time_ms == 0.0 and fps > 0:
                        current_time_ms = (current_frame / fps) * 1000.0
                
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Current video position: {current_time_ms}ms (frame {current_frame}, fps {fps})")
                
                # Extract funscript from current position onwards (your suggested approach!)
                # This creates a new funscript where the current video position becomes time 0
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Creating time-extracted funscript starting from {current_time_ms}ms")
                
                # Get funscript actions using the same method as detection
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Getting funscript actions")
                if not hasattr(self.app, 'funscript_processor') or not self.app.funscript_processor:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No funscript processor found")
                    self.handy_preparing = False
                    return
                    
                fs_proc = self.app.funscript_processor
                funscript_obj = fs_proc.get_funscript_obj()
                if not funscript_obj:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No funscript object found")
                    self.handy_preparing = False
                    return
                
                primary_actions = fs_proc.get_actions('primary')
                secondary_actions = fs_proc.get_actions('secondary')
                
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Retrieved {len(primary_actions)} primary actions, {len(secondary_actions)} secondary actions")
                
                if not primary_actions:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No primary actions available")
                    self.handy_preparing = False
                    return
                
                # Create and save temporary funscript file
                import tempfile
                import json
                import os
                
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Creating time-extracted funscript for Handy")
                
                # Extract actions from current video position onwards
                extracted_primary_actions = []
                for action in primary_actions:
                    action_time = action.get('at', 0)
                    if action_time >= current_time_ms:
                        # Adjust timestamp to start from 0 (current video position becomes time 0)
                        adjusted_action = {
                            'at': int(action_time - current_time_ms),  # Integer timestamps for Handy compatibility
                            'pos': int(action.get('pos', 0))  # Integer positions for Handy compatibility
                        }
                        extracted_primary_actions.append(adjusted_action)
                
                # Do the same for secondary actions if present
                extracted_secondary_actions = []
                if secondary_actions:
                    for action in secondary_actions:
                        action_time = action.get('at', 0)
                        if action_time >= current_time_ms:
                            adjusted_action = {
                                'at': int(action_time - current_time_ms),  # Integer timestamps for Handy compatibility
                                'pos': int(action.get('pos', 0))  # Integer positions for Handy compatibility
                            }
                            extracted_secondary_actions.append(adjusted_action)
                
                # Ensure funscript always starts at time=0 for HSSP compatibility
                if extracted_primary_actions and extracted_primary_actions[0]['at'] > 0:
                    # Interpolate the position at current_time_ms for time=0 baseline
                    baseline_pos = 50  # Default if no data
                    
                    # Find the actions before and after current_time_ms for interpolation
                    prev_action = None
                    next_action = None
                    
                    for i, action in enumerate(primary_actions):
                        action_time = action.get('at', 0)
                        if action_time <= current_time_ms:
                            prev_action = action
                        elif action_time > current_time_ms and next_action is None:
                            next_action = action
                            break
                    
                    # Interpolate position at current_time_ms
                    if prev_action and next_action:
                        # Linear interpolation between two actions
                        t1, p1 = prev_action['at'], prev_action['pos']
                        t2, p2 = next_action['at'], next_action['pos']
                        
                        # Calculate interpolation factor (0 to 1)
                        if t2 > t1:
                            factor = (current_time_ms - t1) / (t2 - t1)
                            baseline_pos = p1 + (p2 - p1) * factor
                        else:
                            baseline_pos = p1
                    elif prev_action:
                        # Use last known position if no next action
                        baseline_pos = prev_action.get('pos', 50)
                    elif next_action:
                        # Use next position if no previous action
                        baseline_pos = next_action.get('pos', 50)
                    
                    # Insert interpolated baseline action at time=0
                    baseline_action = {'at': 0, 'pos': int(baseline_pos)}
                    extracted_primary_actions.insert(0, baseline_action)
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Added interpolated baseline action at time=0: {baseline_action}")
                
                # Ensure minimum of 2 actions for HSSP compatibility
                if len(extracted_primary_actions) < 2:
                    # Add a hold action at 1000ms with same position
                    if extracted_primary_actions:
                        last_pos = extracted_primary_actions[-1]['pos']
                    else:
                        last_pos = 50  # Default middle position
                        extracted_primary_actions.append({'at': 0, 'pos': last_pos})
                    
                    hold_action = {'at': 1000, 'pos': last_pos}
                    extracted_primary_actions.append(hold_action)
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Added hold action for HSSP minimum requirement: {hold_action}")
                
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Extracted {len(extracted_primary_actions)} primary actions starting from time 0")
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Original video time {current_time_ms}ms now maps to funscript time 0ms")
                
                if not extracted_primary_actions:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No actions found after current video position - video may be at end")
                    self.handy_preparing = False
                    return
                
                # Show sample extracted actions for debugging
                if extracted_primary_actions:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" First extracted action: {extracted_primary_actions[0]}")
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Last extracted action: {extracted_primary_actions[-1]}")
                    
                    # Get extracted funscript duration
                    last_action_time = max(action.get('at', 0) for action in extracted_primary_actions)
                    first_action_time = min(action.get('at', 0) for action in extracted_primary_actions)
                    funscript_duration_ms = last_action_time - first_action_time
                    
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" First action: at={primary_actions[0].get('at')}ms, pos={primary_actions[0].get('pos')}")
                    if len(primary_actions) > 1:
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Last action: at={last_action_time}ms")
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Funscript duration: {funscript_duration_ms}ms ({funscript_duration_ms/1000:.1f}s)")
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Start time: {current_time_ms}ms")
                    
                    # Check if start time is within funscript range
                    if current_time_ms > last_action_time:
                        _handy_debug_logger.warning(f"Start time ({current_time_ms}ms) is AFTER last action ({last_action_time}ms) - HSSP play may fail")
                    elif current_time_ms < first_action_time:
                        _handy_debug_logger.warning(f"Start time ({current_time_ms}ms) is BEFORE first action ({first_action_time}ms)")
                    elif _handy_debug_logger.isEnabledFor(logging.DEBUG):
                        _handy_debug_logger.debug(f"Start time is within funscript range ({first_action_time}ms - {last_action_time}ms)")
                    
                    # Find actions around the current video position
                    nearby_actions = [a for a in primary_actions if abs(a.get('at', 0) - current_time_ms) < 5000]
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Actions within 5s of current position ({current_time_ms}ms): {len(nearby_actions)}")
                    if nearby_actions:
                        for i, action in enumerate(nearby_actions[:3]):
                            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Nearby action {i+1}: at={action.get('at')}ms, pos={action.get('pos')}")
                
                # Use extracted actions that start from time 0 (current video position)
                funscript_data = {
                    "actions": extracted_primary_actions,
                    "inverted": False,
                    "range": 90,
                    "version": "1.0"
                }
                
                # Save to temporary file
                temp_dir = tempfile.gettempdir()
                temp_filename = f"handy_stream_{int(current_time_ms)}.funscript"
                temp_path = os.path.join(temp_dir, temp_filename)
                
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Saving funscript to: {temp_path}")
                with open(temp_path, 'w') as f:
                    json.dump(funscript_data, f, indent=2)
                
                self.handy_last_funscript_path = temp_path
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Funscript file saved successfully")
                
                # Start async workflow
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Creating new event loop")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                device_manager = self.app.device_manager
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Device manager available: {device_manager is not None}")
                
                # Prepare Handy devices with extracted (time-shifted) actions
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Calling prepare_handy_for_video_playback with extracted actions")
                prepare_success = loop.run_until_complete(
                    device_manager.prepare_handy_for_video_playback(extracted_primary_actions, extracted_secondary_actions)
                )
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Prepare result: {prepare_success}")
                
                if not prepare_success:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Prepare failed, returning")
                    self.handy_preparing = False
                    return
                
                # Wait a moment for upload to complete
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Waiting 2 seconds for upload to complete")
                loop.run_until_complete(asyncio.sleep(2))
                
                # Start synchronized playback from time 0 (since our funscript now starts at 0)
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Starting video sync at 0ms (extracted funscript starts from current video position {current_time_ms}ms)")
                start_success = loop.run_until_complete(
                    device_manager.start_handy_video_sync(0.0)  # Always start from 0 with extracted funscript
                )
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Video sync start result: {start_success}")
                
                if start_success:
                    self.handy_streaming_active = True
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Handy streaming activated successfully!")
                    
                    # Auto-start video playback for real-time sync
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Auto-starting video playback for Handy sync")
                    try:
                        # Check if video is not currently playing
                        is_currently_playing = (self.app.processor and 
                                              self.app.processor.is_processing and 
                                              not self.app.processor.pause_event.is_set())
                        
                        if not is_currently_playing:
                            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video not playing, starting playback")
                            # Start video playback using the same method as the play button
                            if hasattr(self.app, 'event_handlers'):
                                self.app.event_handlers.handle_playback_control("play_pause")
                                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video playback started via event handler")
                            else:
                                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No event handlers available")
                        else:
                            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video already playing")
                            
                    except Exception as playback_error:
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Failed to auto-start video playback: {playback_error}")
                    
                    if hasattr(self.app, 'logger'):
                        self.app.logger.info(f"Handy streaming started at {current_time_ms:.1f}ms")
                else:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video sync start failed")
                
                self.handy_preparing = False
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" handy_preparing set to False, streaming setup complete")
                
            except Exception as e:
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Exception in streaming setup: {e}")
                import traceback
                traceback.print_exc()
                self.handy_preparing = False
                if hasattr(self.app, 'logger'):
                    self.app.logger.error(f"Failed to start Handy streaming: {e}")
            finally:
                if loop is not None:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Closing event loop")
                    loop.close()
                else:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No loop to close")
        
        # Start in background thread
        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Creating background thread for Handy streaming")
        thread = threading.Thread(target=start_streaming_async, name="HandyStreamStart", daemon=True)
        thread.start()
        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Background thread started")
    

    def _stop_handy_streaming(self):
        """Stop Handy streaming and clean up."""
        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" _stop_handy_streaming() called")
        
        # Restore original video speed mode
        if (self.saved_processing_speed_mode is not None and 
            hasattr(self.app, 'app_state_ui') and 
            hasattr(self.app.app_state_ui, 'selected_processing_speed_mode')):
            self.app.app_state_ui.selected_processing_speed_mode = self.saved_processing_speed_mode
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Restored video speed to {self.saved_processing_speed_mode.value}")
            self.saved_processing_speed_mode = None
        
        try:
            # Stop Handy device streaming
            if hasattr(self.app, 'device_manager') and self.app.device_manager:
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Stopping Handy device streaming")
                self.app.device_manager.stop_handy_streaming()
            else:
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No device manager available for stopping")
            
            # Stop video playback
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Stopping video playback")
            try:
                if hasattr(self.app, 'processor') and self.app.processor:
                    is_currently_playing = (self.app.processor.is_processing and 
                                          not self.app.processor.pause_event.is_set())
                    
                    if is_currently_playing:
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video is playing, pausing it")
                        if hasattr(self.app, 'event_handlers'):
                            self.app.event_handlers.handle_playback_control("play_pause")
                            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video playback paused via event handler")
                        else:
                            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No event handlers available for stopping video")
                    else:
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Video was not playing")
                else:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" No video processor available")
                    
            except Exception as video_stop_error:
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Failed to stop video playback: {video_stop_error}")
            
            # Update streaming state
            self.handy_streaming_active = False
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" handy_streaming_active set to False")
            
            # Clean up temporary funscript file
            if self.handy_last_funscript_path:
                try:
                    import os
                    if os.path.exists(self.handy_last_funscript_path):
                        os.remove(self.handy_last_funscript_path)
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Removed temporary funscript file: {self.handy_last_funscript_path}")
                    else:
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Temporary funscript file not found: {self.handy_last_funscript_path}")
                except Exception as cleanup_error:
                    if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Failed to clean up funscript file: {cleanup_error}")
                self.handy_last_funscript_path = None
            
            if hasattr(self.app, 'logger'):
                self.app.logger.info("Handy streaming stopped")
                
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Handy streaming stop complete")
                
        except Exception as e:
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Error stopping Handy streaming: {e}")
    

    def _resync_handy_after_seek(self):
        """Resynchronize Handy device after video seek."""
        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" _resync_handy_after_seek() called")
        
        if not self.handy_streaming_active:
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Handy streaming not active, skipping resync")
            return
        
        try:
            # Stop current streaming first
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Stopping current Handy streaming for resync")
            if hasattr(self.app, 'device_manager') and self.app.device_manager:
                # Just stop the playback, not the entire streaming session
                import asyncio
                
                def stop_and_resync():
                    loop = None
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # Stop current HSSP playback
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Stopping HSSP playback for resync")
                        loop.run_until_complete(self.app.device_manager.stop_handy_playback())
                        
                        # Wait a brief moment for stop to complete
                        loop.run_until_complete(asyncio.sleep(0.5))
                        
                        # Restart with new position
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Restarting Handy streaming from new position")
                        self._start_handy_streaming()
                        
                    except Exception as e:
                        if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Error during resync: {e}")
                        if hasattr(self.app, 'logger'):
                            self.app.logger.error(f"Failed to resync Handy after seek: {e}")
                    finally:
                        if loop is not None:
                            loop.close()
                
                # Run resync in background thread
                import threading
                thread = threading.Thread(target=stop_and_resync, name="HandyResync", daemon=True)
                thread.start()
                if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(" Resync thread started")
                
        except Exception as e:
            if _handy_debug_logger.isEnabledFor(logging.DEBUG): _handy_debug_logger.debug(f" Exception in Handy resync: {e}")
            if hasattr(self.app, 'logger'):
                self.app.logger.error(f"Failed to resync Handy after seek: {e}")

