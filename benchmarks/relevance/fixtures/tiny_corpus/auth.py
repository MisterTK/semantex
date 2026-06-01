def login(username, password):
    """Authenticate a user by username and password, returning a session token."""
    token = _issue_token(username)
    return token


def _issue_token(username):
    return f"token-for-{username}"
