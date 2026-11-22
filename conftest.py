"""Make pytest fail quickly if collection or tests contact a broker or site."""

import socket

def _blocked_connect(self, address):
    raise AssertionError(f"Tests must mock network connections: {address!r}")


def pytest_sessionstart(session):
    # Covers collection too: an accidentally named integration script must
    # never send an order or notification simply by being imported.
    session._original_socket_connect = socket.socket.connect
    session._original_socket_connect_ex = socket.socket.connect_ex
    socket.socket.connect = _blocked_connect
    socket.socket.connect_ex = _blocked_connect


def pytest_sessionfinish(session, exitstatus):
    socket.socket.connect = session._original_socket_connect
    socket.socket.connect_ex = session._original_socket_connect_ex
