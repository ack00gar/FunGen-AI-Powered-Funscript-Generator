import os
from send2trash import send2trash
import logging

class GeneratedFileManager:
    def __init__(self, output_folder, logger=None):
        self.output_folder = output_folder
        self.logger = logger or logging.getLogger(__name__)
        self.file_tree = {}
        self.total_size_mb = 0
        self._scan_files()

    def _scan_files(self):
        self.file_tree = {}
        total_size_bytes = 0
        if not os.path.isdir(self.output_folder):
            return

        for video_dir_name in os.listdir(self.output_folder):
            video_dir_path = os.path.join(self.output_folder, video_dir_name)
            if os.path.isdir(video_dir_path):
                files_in_dir = []
                folder_total_size_bytes = 0
                for filename in os.listdir(video_dir_path):
                    file_path = os.path.join(video_dir_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            size_bytes = os.path.getsize(file_path)
                            total_size_bytes += size_bytes
                            folder_total_size_bytes += size_bytes
                            files_in_dir.append(
                                {"name": filename, "path": file_path, "size_mb": size_bytes / (1024 * 1024)}
                            )
                        except OSError:
                            continue

                if files_in_dir or not os.listdir(video_dir_path):
                    self.file_tree[video_dir_name] = {
                        "path": video_dir_path,
                        "files": sorted(files_in_dir, key=lambda x: x['name']),
                        "total_size_mb": folder_total_size_bytes / (1024 * 1024)
                    }
        self.total_size_mb = total_size_bytes / (1024 * 1024)

    def delete_file(self, file_path):
        if file_path and os.path.exists(file_path):
            try:
                send2trash(file_path)
                return True
            except Exception as e:
                self.logger.error(f"ERROR deleting file: {file_path}: {e}")
                return False
        return False

    def delete_folder(self, folder_path, include_funscript_files=True):
        if folder_path and os.path.isdir(folder_path):
            try:
                if include_funscript_files:
                    send2trash(folder_path)
                else:
                    # Delete all except .funscript files in the folder
                    for dirpath, dirnames, filenames in os.walk(folder_path):
                        for filename in filenames:
                            if filename.endswith('.funscript'):
                                continue
                            file_path = os.path.join(dirpath, filename)
                            try:
                                send2trash(file_path)
                            except Exception as e:
                                self.logger.error(f"ERROR deleting file: {file_path}: {e}")
                    # Remove empty folders (except those containing .funscript files)
                    for dirpath, dirnames, filenames in os.walk(folder_path, topdown=False):
                        if all(f.endswith('.funscript') for f in filenames):
                            continue
                        try:
                            if not os.listdir(dirpath):
                                send2trash(dirpath)
                        except Exception:
                            pass
                return True
            except Exception as e:
                self.logger.error(f"ERROR deleting folder: {folder_path}: {e}")
                return False
        return False

    def delete_all(self, include_funscript_files=True):
        if not os.path.isdir(self.output_folder):
            return False
        try:
            for entry in os.listdir(self.output_folder):
                entry_path = os.path.join(self.output_folder, entry)
                if include_funscript_files:
                    send2trash(entry_path)
                else:
                    if os.path.isdir(entry_path):
                        self.delete_folder(entry_path, include_funscript_files=False)
                    elif os.path.isfile(entry_path) and not entry_path.endswith('.funscript'):
                        send2trash(entry_path)
            return True
        except Exception as e:
            self.logger.error(f"ERROR deleting all files in {self.output_folder}: {e}")
            return False 

    def get_sorted_file_tree(self, sort_by='name'):
        """
        Returns a sorted list of (video_dir, dir_data) tuples based on sort_by ('name' or 'size').
        """
        folder_items = list(self.file_tree.items())
        if sort_by == 'size':
            folder_items.sort(key=lambda item: item[1]['total_size_mb'], reverse=True)
        else:
            folder_items.sort(key=lambda item: item[0].lower())
        return folder_items

    @property
    def total_size(self):
        """Returns the total size in MB of all files managed."""
        return self.total_size_mb 