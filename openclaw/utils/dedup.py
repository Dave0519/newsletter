class DedupDB:
    def __init__(self, path: str):
        self.path = path
    def has(self, *_args, **_kwargs):
        return False
    def add(self, *_args, **_kwargs):
        return None
