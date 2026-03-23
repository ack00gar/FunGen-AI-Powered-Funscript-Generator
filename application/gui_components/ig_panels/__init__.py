"""Info Graphs panel — composed from focused sub-mixins."""
from application.gui_components.ig_panels._video_info import VideoInfoMixin
from application.gui_components.ig_panels._video_settings import VideoSettingsMixin
from application.gui_components.ig_panels._funscript_info import FunscriptInfoMixin
from application.gui_components.ig_panels._comparison import ComparisonMixin
from application.gui_components.ig_panels._performance import PerformanceMixin
from application.gui_components.ig_panels._developer_perf import DeveloperPerfMixin
from application.gui_components.ig_panels._undo_history import UndoHistoryMixin


class InfoGraphsMixin(
    VideoInfoMixin,
    VideoSettingsMixin,
    FunscriptInfoMixin,
    ComparisonMixin,
    PerformanceMixin,
    DeveloperPerfMixin,
    UndoHistoryMixin,
):
    """Composed from focused sub-mixins for Info Graphs panel rendering."""
    pass
