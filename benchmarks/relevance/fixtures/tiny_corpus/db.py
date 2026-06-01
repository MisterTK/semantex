class ConnectionPool:
    """A pool of reusable database connections."""

    def __init__(self, size):
        self.size = size
        self._connections = []

    def acquire(self):
        return self._connections.pop()
