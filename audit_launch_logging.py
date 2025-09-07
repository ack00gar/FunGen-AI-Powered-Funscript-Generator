#!/usr/bin/env python
"""
Comprehensive audit of FunGen launch logging.
Captures and analyzes all logging during application startup.
"""

import sys
import io
import logging
import time
import threading
from contextlib import redirect_stdout, redirect_stderr

class LaunchLoggingAuditor:
    def __init__(self):
        self.log_entries = []
        self.start_time = None
        self.phase_times = {}
        self.current_phase = "init"
        
    def capture_logs(self):
        """Capture all log output during launch."""
        # Create custom handler to intercept all logs
        class AuditHandler(logging.Handler):
            def __init__(self, auditor):
                super().__init__()
                self.auditor = auditor
                
            def emit(self, record):
                timestamp = time.time() - self.auditor.start_time if self.auditor.start_time else 0
                self.auditor.log_entries.append({
                    'timestamp': timestamp,
                    'phase': self.auditor.current_phase,
                    'level': record.levelname,
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'message': record.getMessage(),
                    'name': record.name
                })
                
        # Install our audit handler at root level
        root_logger = logging.getLogger()
        audit_handler = AuditHandler(self)
        root_logger.addHandler(audit_handler)
        
        return audit_handler
    
    def analyze_logs(self):
        """Analyze captured logs for issues."""
        print("🔍 FUNGEN LAUNCH LOGGING AUDIT RESULTS")
        print("=" * 60)
        
        if not self.log_entries:
            print("❌ No logs captured!")
            return
        
        # Summary stats
        total_logs = len(self.log_entries)
        by_level = {}
        by_phase = {}
        by_module = {}
        redundant_messages = {}
        
        for entry in self.log_entries:
            # Count by level
            by_level[entry['level']] = by_level.get(entry['level'], 0) + 1
            
            # Count by phase  
            by_phase[entry['phase']] = by_phase.get(entry['phase'], 0) + 1
            
            # Count by module
            by_module[entry['module']] = by_module.get(entry['module'], 0) + 1
            
            # Check for redundant messages
            msg = entry['message']
            if msg in redundant_messages:
                redundant_messages[msg] += 1
            else:
                redundant_messages[msg] = 1
        
        print(f"\n📊 SUMMARY:")
        print(f"  Total log entries: {total_logs}")
        print(f"  Launch duration: {self.log_entries[-1]['timestamp']:.2f}s")
        
        print(f"\n📈 BY LOG LEVEL:")
        for level, count in sorted(by_level.items(), key=lambda x: -x[1]):
            pct = (count / total_logs) * 100
            print(f"  {level:8}: {count:3} ({pct:4.1f}%)")
            
        print(f"\n🏗️ BY LAUNCH PHASE:")
        for phase, count in sorted(by_phase.items(), key=lambda x: -x[1]):
            pct = (count / total_logs) * 100
            print(f"  {phase:15}: {count:3} ({pct:4.1f}%)")
            
        print(f"\n🔗 BY MODULE (top 10):")
        top_modules = sorted(by_module.items(), key=lambda x: -x[1])[:10]
        for module, count in top_modules:
            pct = (count / total_logs) * 100
            print(f"  {module:20}: {count:3} ({pct:4.1f}%)")
        
        # Find redundant messages
        redundant = {msg: count for msg, count in redundant_messages.items() if count > 1}
        if redundant:
            print(f"\n⚠️ REDUNDANT MESSAGES ({len(redundant)} unique messages):")
            for msg, count in sorted(redundant.items(), key=lambda x: -x[1])[:10]:
                print(f"  {count}x: {msg[:60]}")
        
        # Find potential issues
        issues = []
        
        # Check for excessive INFO logs
        info_count = by_level.get('INFO', 0)
        if info_count > 20:
            issues.append(f"Excessive INFO logs ({info_count}) - consider reducing verbosity")
            
        # Check for debug logs in production
        debug_count = by_level.get('DEBUG', 0)
        if debug_count > 0:
            issues.append(f"DEBUG logs present ({debug_count}) - may slow launch")
            
        # Check for slow phases
        phase_durations = self.calculate_phase_durations()
        for phase, duration in phase_durations.items():
            if duration > 2.0:  # > 2 seconds
                issues.append(f"Slow launch phase '{phase}': {duration:.2f}s")
        
        if issues:
            print(f"\n🚨 POTENTIAL ISSUES:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"\n✅ No major logging issues detected")
            
        return {
            'total_logs': total_logs,
            'by_level': by_level,
            'by_phase': by_phase,
            'by_module': by_module,
            'redundant': redundant,
            'issues': issues,
            'duration': self.log_entries[-1]['timestamp']
        }
    
    def calculate_phase_durations(self):
        """Calculate how long each phase took."""
        phase_durations = {}
        phase_starts = {}
        
        for entry in self.log_entries:
            phase = entry['phase']
            timestamp = entry['timestamp']
            
            if phase not in phase_starts:
                phase_starts[phase] = timestamp
            
            # Update end time for this phase
            phase_durations[phase] = timestamp - phase_starts[phase]
            
        return phase_durations
    
    def show_timeline(self, limit=20):
        """Show chronological timeline of key log events."""
        print(f"\n⏱️ LAUNCH TIMELINE (showing first {limit} entries):")
        print("-" * 80)
        
        for i, entry in enumerate(self.log_entries[:limit]):
            timestamp = f"{entry['timestamp']:6.2f}s"
            level = f"{entry['level']:7}"
            module = f"{entry['module']:15}"
            message = entry['message'][:50]
            
            print(f"  {timestamp} | {level} | {module} | {message}")
        
        if len(self.log_entries) > limit:
            print(f"  ... ({len(self.log_entries) - limit} more entries)")

def run_audit():
    """Run the launch logging audit."""
    print("🚀 Starting FunGen Launch Logging Audit...")
    
    auditor = LaunchLoggingAuditor()
    auditor.start_time = time.time()
    
    # Capture all logging
    audit_handler = auditor.capture_logs()
    
    try:
        # Simulate launch phases
        print("Phase 1: Bootstrap & Dependency Check")
        auditor.current_phase = "bootstrap"
        
        # Import main to trigger bootstrap logging
        sys.path.insert(0, '/Users/k00gar/PycharmProjects/VR-Funscript-AI-Generator')
        from main import _setup_bootstrap_logger
        _setup_bootstrap_logger()
        
        auditor.current_phase = "dependency_check"
        from application.utils.dependency_checker import check_and_install_dependencies
        check_and_install_dependencies()
        
        print("Phase 2: Application Logic Initialization")
        auditor.current_phase = "app_logic_init"
        from application.logic.app_logic import ApplicationLogic
        app_logic = ApplicationLogic(is_cli=False)
        
        print("Phase 3: GUI Initialization")  
        auditor.current_phase = "gui_init"
        from application.gui_components.app_gui import GUI
        # Note: We don't actually create the GUI as it would open a window
        
        print("Phase 4: Tracker Discovery")
        auditor.current_phase = "tracker_discovery"
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        
        time.sleep(0.1)  # Let any async logging finish
        
    except Exception as e:
        print(f"❌ Error during audit: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Remove our handler
        logging.getLogger().removeHandler(audit_handler)
    
    # Analyze results
    results = auditor.analyze_logs()
    auditor.show_timeline()
    
    return results

if __name__ == "__main__":
    results = run_audit()