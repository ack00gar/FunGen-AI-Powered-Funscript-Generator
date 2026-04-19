from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BatchState:
    videos_data: List[Dict] = field(default_factory=list)
    selected_method_idx_ui: int = 0
    overwrite_mode_ui: int = 0
    set_all_format_idx: int = 1
    copy_funscript_to_video_location_ui: bool = True
    generate_roll_file_ui: bool = True
    apply_ultimate_autotune_ui: bool = True
    adaptive_tuning_ui: bool = True
    save_preprocessed_video_ui: bool = False
    last_overwrite_mode_ui: int = -1
