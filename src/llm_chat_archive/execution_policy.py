from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .models import CollectionExecutionPolicy

_CURRENT_EXECUTION_POLICY: ContextVar[CollectionExecutionPolicy] = ContextVar(
    "current_collection_execution_policy",
    default=CollectionExecutionPolicy(),
)


def get_collection_execution_policy() -> CollectionExecutionPolicy:
    return _CURRENT_EXECUTION_POLICY.get()


@contextmanager
def collection_execution_policy_context(
    policy: CollectionExecutionPolicy,
) -> Iterator[None]:
    token = _CURRENT_EXECUTION_POLICY.set(policy)
    try:
        yield
    finally:
        _CURRENT_EXECUTION_POLICY.reset(token)


__all__ = [
    "collection_execution_policy_context",
    "get_collection_execution_policy",
]
