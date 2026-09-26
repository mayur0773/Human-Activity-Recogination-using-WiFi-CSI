import os
import re
import csv
from datetime import datetime
 
# ==========================================================
# CONFIG
# ==========================================================
 
# Point this at your RAW dataset (the original recordings,
# NOT the processed/feature-extracted output).
RAW_DATASET = r"C:\CSI_Project\Dataset"
 
OUTPUT_MAP = "session_map.csv"
 
# Two recordings are considered the SAME session if the gap
# between their file timestamps is <= this many minutes.
# Only used as a FALLBACK for old-format filenames that don't
# carry an embedded session tag (see below).
SESSION_GAP_MINUTES = 20
 
# New-format filenames look like: Walking_20260815_1830_001.csv
# (activity_YYYYMMDD_HHMM_NNN.csv), written by the updated
# recorder.py. If this pattern matches, we trust the embedded
# session tag directly instead of guessing from timestamps.
NEW_FORMAT_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{4})_(\d+)\.csv$")
 
 
def embedded_session_id(filename):
    """Return 'session_20260815_1830' style ID if the filename
    already carries a real session tag, else None."""
    match = NEW_FORMAT_PATTERN.match(filename)
    if match:
        return f"session_{match.group(2)}"
    return None
 
 
def main():
 
    if not os.path.exists(RAW_DATASET):
        print(f"ERROR: {RAW_DATASET} not found. Update RAW_DATASET.")
        return
 
    records = []
 
    for activity in sorted(os.listdir(RAW_DATASET)):
 
        activity_path = os.path.join(RAW_DATASET, activity)
 
        if not os.path.isdir(activity_path):
            continue
 
        for filename in sorted(os.listdir(activity_path)):
 
            if not filename.endswith(".csv"):
                continue
 
            filepath = os.path.join(activity_path, filename)
            mtime = os.path.getmtime(filepath)
 
            records.append({
                "Activity": activity,
                "Filename": filename,
                "ModifiedTime": mtime,
                "EmbeddedSession": embedded_session_id(filename)
            })
 
    if not records:
        print("ERROR: No CSV files found under RAW_DATASET.")
        return
 
    # ------------------------------------------------------
    # Split into: files that already have a real session tag
    # (new format) vs files that need timestamp reconstruction
    # (old format).
    # ------------------------------------------------------
 
    new_format = [r for r in records if r["EmbeddedSession"] is not None]
    old_format = [r for r in records if r["EmbeddedSession"] is None]
 
    for r in new_format:
        r["SessionID"] = r["EmbeddedSession"]
 
    # Timestamp-cluster only the old-format files, same as before.
    old_format.sort(key=lambda r: r["ModifiedTime"])
 
    session_id = 0
    prev_time = None
    gap_seconds = SESSION_GAP_MINUTES * 60
 
    for r in old_format:
 
        if prev_time is None or (r["ModifiedTime"] - prev_time) > gap_seconds:
            session_id += 1
 
        r["SessionID"] = f"reconstructed_session_{session_id:03d}"
        prev_time = r["ModifiedTime"]
 
    records = new_format + old_format
    records.sort(key=lambda r: r["ModifiedTime"])
 
    # ------------------------------------------------------
    # WRITE MAP
    # ------------------------------------------------------
 
    with open(OUTPUT_MAP, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["Activity", "Filename", "SessionID", "ModifiedTime"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "Activity": r["Activity"],
                "Filename": r["Filename"],
                "SessionID": r["SessionID"],
                "ModifiedTime": datetime.fromtimestamp(
                    r["ModifiedTime"]
                ).strftime("%Y-%m-%d %H:%M:%S")
            })
 
    # ------------------------------------------------------
    # SANITY CHECK OUTPUT
    # ------------------------------------------------------
 
    print(f"\n{OUTPUT_MAP} created with {len(records)} recordings.")
    print(f"  New-format (real session tag) : {len(new_format)} files")
    print(f"  Old-format (timestamp-guessed): {len(old_format)} files\n")
 
    print("Session breakdown (session -> activities it contains):")
    from collections import defaultdict
    sess_activities = defaultdict(set)
    sess_counts = defaultdict(int)
    for r in records:
        sess_activities[r["SessionID"]].add(r["Activity"])
        sess_counts[r["SessionID"]] += 1
 
    for sid in sorted(sess_activities.keys()):
        print(f"  {sid}: {sess_counts[sid]} files, activities={sorted(sess_activities[sid])}")
 
    print(
        "\nCHECK THIS: if every session lists only ONE activity, that "
        "session isn't useful for GroupKFold generalization testing. "
        "New sessions (from the updated recorder.py) should always mix "
        "activities if you record all 4 activities per sitting -- if "
        "they don't, you're still running recorder.py once per activity "
        "instead of once per session covering all activities."
    )
 
 
if __name__ == "__main__":
    main()