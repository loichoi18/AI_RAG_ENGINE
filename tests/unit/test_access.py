"""Unit tests for ACL access rules and the mock user directory."""

from __future__ import annotations

import pytest

from retrieval.access import GROUP_ENGINEERING, GROUP_HR, acl_allows, get_user_acl


def test_unrestricted_chunk_visible_to_everyone() -> None:
    assert acl_allows([], [GROUP_HR]) is True
    assert acl_allows([], None) is True


def test_restricted_chunk_requires_group_intersection() -> None:
    assert acl_allows([GROUP_HR], [GROUP_HR]) is True
    assert acl_allows([GROUP_HR], [GROUP_ENGINEERING]) is False


def test_admin_none_acl_sees_everything() -> None:
    assert acl_allows([GROUP_HR], None) is True


def test_mock_user_acls() -> None:
    assert get_user_acl("admin") is None
    assert get_user_acl("engineering") == [GROUP_ENGINEERING]
    assert get_user_acl("hr") == [GROUP_HR]


def test_unknown_user_raises() -> None:
    with pytest.raises(KeyError):
        get_user_acl("ceo")
