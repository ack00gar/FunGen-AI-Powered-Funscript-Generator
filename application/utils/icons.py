"""Named icon constants from the merged Symbols Nerd Font.

Usage:
    from application.utils.icons import ICON_PLAY, ICON_FOLDER
    imgui.button(f"{ICON_PLAY} Play")

The codepoints come from the Font Awesome / Material / Codicons blocks
included in Symbols Nerd Font. Each symbol inherits the current imgui
text color since the font is monochrome.

If the icon font failed to load, these constants render as fallback
glyphs from the default Latin range (configurable via FALLBACK_ICON).
"""

# Single-glyph constants. Strings contain one Unicode char each.

# --- Playback / transport ---
ICON_PLAY        = "\uf04b"   # fa-play
ICON_PAUSE       = "\uf04c"   # fa-pause
ICON_STOP        = "\uf04d"   # fa-stop
ICON_STEP_FWD    = "\uf051"   # fa-step-forward
ICON_STEP_BACK   = "\uf048"   # fa-step-backward
ICON_FAST_FWD    = "\uf04e"   # fa-forward
ICON_FAST_BACK   = "\uf04a"   # fa-backward
ICON_RECORD      = "\uf111"   # fa-circle
ICON_REWIND      = "\uf049"   # fa-rewind

# --- File / project ---
ICON_FOLDER      = "\uf07b"   # fa-folder
ICON_FOLDER_OPEN = "\uf07c"   # fa-folder-open
ICON_FILE        = "\uf15b"   # fa-file
ICON_SAVE        = "\uf0c7"   # fa-floppy
ICON_OPEN        = "\uf07c"
ICON_DOWNLOAD    = "\uf019"   # fa-download
ICON_UPLOAD      = "\uf093"   # fa-upload
ICON_EXPORT      = "\uf08e"   # fa-external-link
ICON_IMPORT      = "\uf090"   # fa-sign-in

# --- Edit / clipboard ---
ICON_COPY        = "\uf0c5"
ICON_PASTE       = "\uf0ea"
ICON_CUT         = "\uf0c4"
ICON_UNDO        = "\uf0e2"
ICON_REDO        = "\uf01e"
ICON_TRASH       = "\uf1f8"
ICON_EDIT        = "\uf044"
ICON_PENCIL      = "\uf040"
ICON_PLUS        = "\uf067"
ICON_MINUS       = "\uf068"
ICON_CLOSE       = "\uf00d"
ICON_CHECK       = "\uf00c"

# --- UI / nav ---
ICON_SETTINGS    = "\uf013"   # fa-gear
ICON_TOOLS       = "\uf7d9"   # fa-tools
ICON_SEARCH      = "\uf002"
ICON_REFRESH     = "\uf021"   # fa-sync
ICON_RELOAD      = "\uf2f9"
ICON_FILTER      = "\uf0b0"
ICON_LIST        = "\uf03a"
ICON_GRID        = "\uf00a"
ICON_MENU        = "\uf0c9"
ICON_CHEVRON_L   = "\uf053"
ICON_CHEVRON_R   = "\uf054"
ICON_CHEVRON_U   = "\uf077"
ICON_CHEVRON_D   = "\uf078"
ICON_ARROW_L     = "\uf060"
ICON_ARROW_R     = "\uf061"
ICON_ARROW_U     = "\uf062"
ICON_ARROW_D     = "\uf063"
ICON_EYE         = "\uf06e"
ICON_EYE_OFF     = "\uf070"

# --- Status ---
ICON_INFO        = "\uf05a"
ICON_WARN        = "\uf071"
ICON_ERROR       = "\uf057"
ICON_SUCCESS     = "\uf058"   # check-circle
ICON_QUESTION    = "\uf059"

# --- Content ---
ICON_VIDEO       = "\uf03d"
ICON_FILM        = "\uf008"
ICON_CAMERA      = "\uf030"
ICON_IMAGE       = "\uf03e"
ICON_BOOKMARK    = "\uf02e"
ICON_TAG         = "\uf02b"
ICON_FLAG        = "\uf024"
ICON_STAR        = "\uf005"
ICON_HEART       = "\uf004"
ICON_CHAPTER     = "\uf02d"   # book
ICON_CLOCK       = "\uf017"
ICON_CHART       = "\uf080"
ICON_WAVE        = "\uf083"   # bar-chart

# --- Lock / security ---
ICON_LOCK        = "\uf023"
ICON_UNLOCK      = "\uf09c"
ICON_KEY         = "\uf084"

# --- Device / network ---
ICON_WIFI        = "\uf1eb"
ICON_BLUETOOTH   = "\uf293"
ICON_USB         = "\uf287"
ICON_CPU         = "\uf2db"   # microchip
ICON_GPU         = "\uf109"   # desktop
ICON_RAM         = "\uf538"   # memory
ICON_DISK        = "\uf0a0"
ICON_CLOUD       = "\uf0c2"
ICON_LINK        = "\uf0c1"

# --- Misc ---
ICON_BOLT        = "\uf0e7"
ICON_FIRE        = "\uf06d"
ICON_MAGIC       = "\uf0d0"
ICON_ROBOT       = "\uf544"
ICON_BRAIN       = "\uf5dc"
ICON_LIGHTBULB   = "\uf0eb"
ICON_PUZZLE      = "\uf12e"
ICON_PLUG        = "\uf1e6"
