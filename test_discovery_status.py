#!/usr/bin/env python3
"""
Test if tracker discovery is working properly to see if we can remove fallbacks
"""

import logging
logging.basicConfig(level=logging.INFO)

print("🔍 Testing whether tracker discovery is reliable enough to remove fallbacks...")

try:
    # Test the real discovery system
    from application.gui_components.tracker_discovery_ui import TrackerDiscoveryUI
    
    class MockApp:
        def __init__(self):
            self.logger = logging.getLogger("TestApp")
    
    app = MockApp()
    discovery = TrackerDiscoveryUI(app)
    
    print("\n1. Testing Live Tracker Manager availability:")
    if discovery.live_tracker_manager:
        print("✅ Live tracker manager is available")
        
        # Test dynamic discovery
        live_trackers = discovery.discover_live_trackers()
        print(f"✅ Dynamic discovery found {len(live_trackers)} live trackers")
        
        # Test if they all have proper enum mappings
        valid_enums = [t for t in live_trackers if t.legacy_enum_value is not None]
        print(f"✅ {len(valid_enums)} have valid enum mappings")
        
        if len(valid_enums) == len(live_trackers):
            print("✅ ALL trackers have valid enum mappings - fallback may not be needed!")
        else:
            print("⚠️ Some trackers missing enum mappings - fallback still needed")
            
    else:
        print("❌ Live tracker manager NOT available - fallback system IS needed")
        
        # Test fallback
        fallback_trackers = discovery._get_fallback_live_trackers()
        print(f"📋 Fallback provides {len(fallback_trackers)} trackers")
    
    print("\n2. Testing Offline Tracker Discovery:")
    offline_trackers = discovery.discover_offline_trackers()
    print(f"📋 Found {len(offline_trackers)} offline trackers")
    
    print("\n3. Testing UI Integration:")
    from application.gui_components.control_panel_ui import ControlPanelUI
    
    control_panel = ControlPanelUI(app)
    
    # Test if it uses dynamic discovery or fallback
    modes_display, modes_enum, discovered_trackers = control_panel._get_tracker_lists_for_ui(simple_mode=True)
    
    print(f"✅ UI gets {len(modes_display)} trackers")
    if discovered_trackers:
        print("✅ Using dynamic discovery in UI")
    else:
        print("📋 Using fallback in UI")
    
    print("\n4. Summary:")
    
    can_remove_fallback = (
        discovery.live_tracker_manager is not None and
        len(live_trackers) > 0 and
        all(t.legacy_enum_value is not None for t in live_trackers)
    )
    
    if can_remove_fallback:
        print("🎉 RECOMMENDATION: Fallback systems can be REMOVED")
        print("   - Dynamic discovery is working reliably")
        print("   - All trackers have valid enum mappings")
        print("   - Live tracker manager is available")
    else:
        print("⚠️ RECOMMENDATION: Keep fallback systems for now")
        print("   - Discovery system may not be fully reliable yet")
        
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()