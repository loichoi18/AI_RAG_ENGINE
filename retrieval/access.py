"""Access-control helpers and mock user directory.

``acl_allows`` is the single definition of the access rule, reused by retrievers
that filter in Python (BM25) and by tests, so the semantics match the Qdrant
payload filter exactly: a chunk is visible if it has no ACL (unrestricted) or if
its ACL shares at least one identifier with the caller's.

A small mock user directory (admin / engineering / hr) provides ACLs for demos
and tests. ``admin`` maps to ``None`` — meaning *no filter*, i.e. full
visibility — which is how an unrestricted query is expressed end to end.
"""

from __future__ import annotations

from collections.abc import Sequence

# Group identifiers used throughout demos/tests.
GROUP_ENGINEERING = "group:engineering"
GROUP_HR = "group:hr"

# Mock users. ``None`` => unrestricted (admin sees everything).
_MOCK_USER_ACLS: dict[str, list[str] | None] = {
    "admin": None,
    "engineering": [GROUP_ENGINEERING],
    "hr": [GROUP_HR],
}


def acl_allows(chunk_acl: Sequence[str], caller_acl: Sequence[str] | None) -> bool:
    """Return whether a caller may access a chunk.

    Parameters
    ----------
    chunk_acl:
        The chunk's ACL (empty = unrestricted).
    caller_acl:
        The caller's identifiers, or ``None`` for an unrestricted (admin) query.
    """
    if caller_acl is None:
        return True
    if not chunk_acl:
        return True
    return bool(set(chunk_acl) & set(caller_acl))


def get_user_acl(username: str) -> list[str] | None:
    """Return the ACL for a mock user (``None`` = unrestricted).

    Raises
    ------
    KeyError
        If the username is not in the mock directory.
    """
    if username not in _MOCK_USER_ACLS:
        raise KeyError(f"Unknown mock user '{username}'. Known: {sorted(_MOCK_USER_ACLS)}")
    return _MOCK_USER_ACLS[username]
