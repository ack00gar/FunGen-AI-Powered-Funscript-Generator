#!/usr/bin/env python3
"""
Chapter detection helpers — shared logic for position classification and chapter building.

Used by both the VR Hybrid Chapter-Aware tracker and the Chapter Maker.
All functions are stateless and operate on plain dicts/lists.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger("ChapterDetection")

# ── Constants ──────────────────────────────────────────────────────────────────

CONTACT_IOU_THRESHOLD = 0.05

# Position type mapping from YOLO contact class
CONTACT_TO_POSITION = {
    'pussy': 'Cowgirl / Missionary',
    'butt': 'Rev. Cowgirl / Doggy',
    'anus': 'Rev. Cowgirl / Doggy',
    'face': 'Blowjob',
    'hand': 'Handjob',
    'breast': 'Boobjob',
    'foot': 'Footjob',
}

# Priority order for contact resolution (higher = preferred)
CONTACT_PRIORITY = {
    'pussy': 10, 'butt': 9, 'anus': 8, 'face': 7,
    'hand': 5, 'breast': 4, 'foot': 3, 'navel': 1,
}

# Position types that represent active content
ACTIVE_POSITIONS = {
    'Cowgirl / Missionary', 'Rev. Cowgirl / Doggy', 'Blowjob',
    'Handjob', 'Boobjob', 'Footjob',
}

# Chapter building parameters
VOTE_WINDOW_S = 20.0       # Centered smoothing window (seconds)
MIN_ACTIVE_PCT = 0.20      # 20% non-NR frames → classify as active
MIN_CHAPTER_S = 20.0       # Minimum chapter duration (seconds)
MERGE_GAP_S = 60.0         # Merge same-type chapters within this gap
NR_ABSORB_MAX_S = 120.0    # Absorb NR gaps shorter than this
TRIM_TOLERANCE_S = 5.0     # Tolerance when trimming to active frames


# ── Detection Parsing ──────────────────────────────────────────────────────────

def parse_detections(det_objs) -> Tuple[Optional[Dict], List[Dict]]:
    """Parse YOLO detection objects into penis box + contact boxes."""
    penis_box = None
    other_boxes = []
    for d in det_objs:
        if d.class_name == 'penis':
            if penis_box is None or d.confidence > penis_box['conf']:
                penis_box = {'box': d.bbox, 'conf': d.confidence}
        elif d.class_name in CONTACT_TO_POSITION:
            other_boxes.append({
                'box': d.bbox,
                'class': d.class_name,
                'conf': d.confidence,
            })
    return penis_box, other_boxes


def build_contact_info(other_boxes: List[Dict], frame_size: int = 640) -> List[Dict]:
    """Build spatial contact info for a frame's detections."""
    contacts = []
    for ob in other_boxes:
        bx1, by1, bx2, by2 = ob['box']
        contacts.append({
            'class': ob['class'],
            'box': ob['box'],
            'conf': ob['conf'],
            'norm_cy': ((by1 + by2) / 2.0) / frame_size,
            'norm_area': ((bx2 - bx1) * (by2 - by1)) / (frame_size * frame_size),
        })
    return contacts


# ── Position Classification ────────────────────────────────────────────────────

def classify_frame_position(penis_box: Dict, other_boxes: List[Dict]) -> str:
    """Classify a frame's position based on penis-body contact."""
    px1, py1, px2, py2 = penis_box['box']

    contacts = []
    for other in other_boxes:
        ox1, oy1, ox2, oy2 = other['box']

        # Calculate IoU
        ix1 = max(px1, ox1)
        iy1 = max(py1, oy1)
        ix2 = min(px2, ox2)
        iy2 = min(py2, oy2)

        if ix1 < ix2 and iy1 < iy2:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            area_p = (px2 - px1) * (py2 - py1)
            area_o = (ox2 - ox1) * (oy2 - oy1)
            union = area_p + area_o - intersection
            iou = intersection / union if union > 0 else 0

            if iou > CONTACT_IOU_THRESHOLD:
                contacts.append(other)
        else:
            # Proximity check (within combined diagonal distance)
            penis_w = px2 - px1
            penis_h = py2 - py1
            diag = np.sqrt(penis_w**2 + penis_h**2)

            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
            dist = np.sqrt((pcx - ocx)**2 + (pcy - ocy)**2)

            o_diag = np.sqrt((ox2 - ox1)**2 + (oy2 - oy1)**2)
            if dist < (diag + o_diag) * 0.5:
                contacts.append(other)

    if not contacts:
        return 'Close up'

    best_contact = max(contacts, key=lambda c: CONTACT_PRIORITY.get(c['class'], 0))
    return CONTACT_TO_POSITION.get(best_contact['class'], 'Not Relevant')


def classify_no_penis(other_boxes: List[Dict], frame_h: int) -> str:
    """Classify position when penis is not visible but contact boxes are."""
    GENITAL_CLASSES = {'pussy', 'butt', 'anus'}

    genital_boxes = [b for b in other_boxes if b['class'] in GENITAL_CLASSES]
    if genital_boxes:
        best = max(genital_boxes, key=lambda b: (
            CONTACT_PRIORITY.get(b['class'], 0),
            (b['box'][2] - b['box'][0]) * (b['box'][3] - b['box'][1])
        ))
        return CONTACT_TO_POSITION.get(best['class'], 'Not Relevant')

    scored = []
    for b in other_boxes:
        cls = b['class']
        if cls not in CONTACT_TO_POSITION:
            continue
        box = b['box']
        area = (box[2] - box[0]) * (box[3] - box[1])
        priority = CONTACT_PRIORITY.get(cls, 0)
        area_factor = area / (frame_h * frame_h) * 100
        score = priority * (1.0 + area_factor)
        scored.append((score, cls))

    if scored:
        best_cls = max(scored, key=lambda x: x[0])[1]
        return CONTACT_TO_POSITION.get(best_cls, 'Not Relevant')

    return 'Not Relevant'


def classify_segment_spatial(start_frame: int, end_frame: int,
                             frame_contact_info: Dict) -> Optional[str]:
    """Reclassify a segment using spatial statistics of contact boxes."""
    class_counts = Counter()
    class_y_sum = {}
    class_area_sum = {}
    total_contacts = 0

    for fid, contacts in frame_contact_info.items():
        if fid < start_frame or fid > end_frame:
            continue
        for c in contacts:
            cls = c['class']
            if cls not in CONTACT_TO_POSITION:
                continue
            class_counts[cls] += 1
            class_y_sum[cls] = class_y_sum.get(cls, 0.0) + c['norm_cy']
            class_area_sum[cls] = class_area_sum.get(cls, 0.0) + c['norm_area']
            total_contacts += 1

    if total_contacts < 3:
        return None

    class_stats = {}
    for cls, count in class_counts.items():
        class_stats[cls] = {
            'count': count,
            'freq': count / total_contacts,
            'mean_y': class_y_sum[cls] / count,
            'mean_area': class_area_sum[cls] / count,
        }

    has_pussy = 'pussy' in class_stats
    has_butt = 'butt' in class_stats or 'anus' in class_stats
    has_face = 'face' in class_stats
    has_breast = 'breast' in class_stats
    has_hand = 'hand' in class_stats

    if has_pussy:
        return 'Cowgirl / Missionary'
    if has_butt:
        butt_cls = 'butt' if 'butt' in class_stats else 'anus'
        if class_stats[butt_cls]['freq'] > 0.2 or class_stats[butt_cls]['mean_area'] > 0.02:
            return 'Rev. Cowgirl / Doggy'
    if has_face and has_breast:
        if class_stats['face']['mean_y'] < class_stats['breast']['mean_y']:
            return 'Cowgirl / Missionary'
    if has_face and not has_breast and not has_pussy and not has_butt:
        if class_stats['face']['mean_area'] > 0.01 and class_stats['face']['mean_y'] > 0.4:
            return 'Blowjob'
    if has_hand and not has_face:
        return 'Handjob'

    if class_counts:
        best_cls = class_counts.most_common(1)[0][0]
        return CONTACT_TO_POSITION.get(best_cls)

    return None


# ── Chapter Building ───────────────────────────────────────────────────────────

def build_chapters(frame_positions: Dict[int, str], fps: float,
                   total_frames: int, frame_skip: int,
                   penis_frames: Optional[set] = None,
                   frame_contact_info: Optional[Dict] = None) -> List[Dict]:
    """Build chapters from sparse frame-level position votes.

    Pipeline:
    1. Smooth per-frame positions with centered voting window
    2. Build raw chapters from smoothed sequence
    3. Merge short chapters into neighbors
    4. Merge same-position chapters within gap threshold
    5. Absorb short NR gaps between active chapters
    6. Trim chapter boundaries to confirmed-active frames
    7. Reclassify chapters using spatial contact features
    8. Final merge of same-type neighbors
    """
    if not frame_positions:
        return [{'start_frame': 0, 'end_frame': total_frames - 1,
                 'position': 'Unknown'}]

    sorted_frames = sorted(frame_positions.items())
    frame_ids = np.array([f[0] for f in sorted_frames])
    frame_times = frame_ids / fps
    positions = [f[1] for f in sorted_frames]
    vote_window_half = VOTE_WINDOW_S / 2.0

    # Step 1: Smooth with centered majority vote
    smoothed = []
    for i, (fid, pos) in enumerate(sorted_frames):
        t = fid / fps
        mask = (frame_times >= t - vote_window_half) & (frame_times <= t + vote_window_half)
        window_pos = [positions[j] for j in range(len(positions)) if mask[j]]
        counts = Counter(window_pos)
        total = len(window_pos)
        nr_count = counts.get('Not Relevant', 0) + counts.get('Close up', 0)
        active_pct = 1 - nr_count / total

        if active_pct >= MIN_ACTIVE_PCT:
            non_nr = {k: v for k, v in counts.items() if k not in ('Not Relevant', 'Close up')}
            smoothed.append(max(non_nr, key=non_nr.get) if non_nr else 'Not Relevant')
        else:
            smoothed.append('Not Relevant')

    # Step 2: Build raw chapters
    raw_chapters = []
    cur_pos = smoothed[0]
    cur_start = sorted_frames[0][0]
    for i in range(1, len(sorted_frames)):
        if smoothed[i] != cur_pos:
            raw_chapters.append({
                'start_frame': cur_start,
                'end_frame': sorted_frames[i - 1][0],
                'position': cur_pos,
            })
            cur_pos = smoothed[i]
            cur_start = sorted_frames[i][0]
    raw_chapters.append({
        'start_frame': cur_start,
        'end_frame': total_frames - 1,
        'position': cur_pos,
    })

    # Step 3: Merge short chapters
    min_frames = int(MIN_CHAPTER_S * fps)
    merged = _merge_short_chapters(raw_chapters, min_frames)

    # Step 4: Merge same-position close together
    merge_gap_frames = int(MERGE_GAP_S * fps)
    merged = _merge_adjacent_same_position(merged, merge_gap_frames)

    # Step 5: Absorb short NR gaps between active chapters
    nr_absorb_frames = int(NR_ABSORB_MAX_S * fps)
    changed = True
    while changed:
        changed = False
        new_merged = []
        i = 0
        while i < len(merged):
            ch = merged[i]
            dur = ch['end_frame'] - ch['start_frame']
            is_nr = ch['position'] in ('Not Relevant', 'Close up')
            if (is_nr and dur < nr_absorb_frames and
                i > 0 and i < len(merged) - 1 and
                new_merged and new_merged[-1]['position'] not in ('Not Relevant', 'Close up') and
                merged[i + 1]['position'] not in ('Not Relevant', 'Close up')):
                prev_dur = new_merged[-1]['end_frame'] - new_merged[-1]['start_frame']
                next_dur = merged[i + 1]['end_frame'] - merged[i + 1]['start_frame']
                if prev_dur >= next_dur:
                    new_merged[-1]['end_frame'] = ch['end_frame']
                else:
                    merged[i + 1]['start_frame'] = ch['start_frame']
                changed = True
                i += 1
                continue
            new_merged.append(dict(ch))
            i += 1
        merged = new_merged

    # Merge same-type adjacent (before trim/reclassify)
    result = [merged[0]] if merged else []
    for ch in merged[1:]:
        if result[-1]['position'] == ch['position']:
            result[-1]['end_frame'] = ch['end_frame']
        else:
            result.append(dict(ch))

    # Step 6: Trim active chapter boundaries to confirmed-active frames
    confirmed_active = set()
    for fid, pos in frame_positions.items():
        if pos not in ('Not Relevant', 'Close up'):
            if fid in (penis_frames or set()):
                confirmed_active.add(fid)

    if confirmed_active:
        trim_tolerance_frames = int(TRIM_TOLERANCE_S * fps)
        sorted_active = sorted(confirmed_active)

        for ch in result:
            if ch['position'] in ('Not Relevant', 'Close up'):
                continue
            ch_active = [f for f in sorted_active
                         if ch['start_frame'] <= f <= ch['end_frame']]
            if not ch_active:
                continue
            ch_sparse = [f for f in frame_positions
                         if ch['start_frame'] <= f <= ch['end_frame']]
            if len(ch_active) / max(1, len(ch_sparse)) < 0.30:
                continue
            first_active = ch_active[0]
            last_active = ch_active[-1]
            new_start = max(ch['start_frame'], first_active - trim_tolerance_frames)
            new_end = min(ch['end_frame'], last_active + trim_tolerance_frames)
            if new_start < new_end:
                ch['start_frame'] = new_start
                ch['end_frame'] = new_end

    # Step 7: Spatial reclassification for chapters with few penis frames
    if frame_contact_info:
        for ch in result:
            if ch['position'] in ('Not Relevant', 'Close up'):
                continue
            ch_penis_count = sum(1 for f in (penis_frames or set())
                                 if ch['start_frame'] <= f <= ch['end_frame'])
            ch_sparse_count = sum(1 for f in frame_positions
                                  if ch['start_frame'] <= f <= ch['end_frame'])
            if ch_sparse_count > 0 and ch_penis_count / ch_sparse_count < 0.15:
                new_pos = classify_segment_spatial(
                    ch['start_frame'], ch['end_frame'], frame_contact_info)
                if new_pos and new_pos != ch['position']:
                    logger.info(f"  Spatial reclassify [{ch['start_frame']}-{ch['end_frame']}]: "
                                f"{ch['position']} → {new_pos}")
                    ch['position'] = new_pos

    # Step 8: Final merge — trimming and reclassification can leave same-type neighbors
    final = [result[0]] if result else []
    for ch in result[1:]:
        if final[-1]['position'] == ch['position']:
            final[-1]['end_frame'] = ch['end_frame']
        else:
            final.append(dict(ch))

    return final


def _merge_short_chapters(chapters: List[Dict], min_frames: int) -> List[Dict]:
    """Merge chapters shorter than min_frames into their neighbors."""
    if len(chapters) <= 1:
        return chapters

    result = [chapters[0]]
    for ch in chapters[1:]:
        duration = ch['end_frame'] - ch['start_frame']
        if duration < min_frames and result:
            result[-1]['end_frame'] = ch['end_frame']
        else:
            result.append(ch)
    return result


def _merge_adjacent_same_position(chapters: List[Dict], gap_frames: int) -> List[Dict]:
    """Merge chapters with same position separated by a short gap."""
    if len(chapters) <= 1:
        return chapters

    result = [chapters[0]]
    for ch in chapters[1:]:
        prev = result[-1]
        gap = ch['start_frame'] - prev['end_frame']
        if prev['position'] == ch['position'] and gap < gap_frames:
            prev['end_frame'] = ch['end_frame']
        else:
            result.append(ch)
    return result
