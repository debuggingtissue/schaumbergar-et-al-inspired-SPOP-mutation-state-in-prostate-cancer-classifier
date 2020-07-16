def get_all_files_in_directory(path):
    return [x for x in path.iterdir() if x.is_file()]
