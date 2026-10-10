"""Tests for metaculus_bot.research.providers helpers."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from metaculus_bot.research import providers
from metaculus_bot.research.providers import choose_provider_with_name, is_asknews_subscription_error


class _FakeForbiddenError(Exception):
    """Stand-in for ``asknews_sdk.errors.ForbiddenError`` — matched by class name."""


# Rename the class attribute so ``type(exc).__name__`` matches the SDK's real
# ForbiddenError class name (the predicate matches on class name substring).
_FakeForbiddenError.__name__ = "ForbiddenError"


class TestIsAsknewsSubscriptionError:
    """is_asknews_subscription_error matches only when BOTH class name and
    message signature match. Narrow matching is load-bearing: we don't want
    generic 403s from unrelated providers to be silenced as off-season noise.
    """

    def test_forbidden_error_with_subscription_code_matches(self) -> None:
        exc = _FakeForbiddenError("403011 - subscription is not currently active")
        assert is_asknews_subscription_error(exc) is True

    def test_forbidden_error_with_subscription_phrase_matches(self) -> None:
        # Alternate message wording — the check accepts either 403011 or the phrase.
        exc = _FakeForbiddenError("subscription is not currently active on this tier")
        assert is_asknews_subscription_error(exc) is True

    def test_forbidden_error_with_unrelated_message_does_not_match(self) -> None:
        # Class name matches but message doesn't: not the subscription-inactive
        # signature. Don't silence — could be a real permission issue.
        exc = _FakeForbiddenError("403000 - rate limit hit")
        assert is_asknews_subscription_error(exc) is False

    def test_generic_runtime_error_with_403_forbidden_does_not_match(self) -> None:
        # Class name doesn't match — must not silence third-party 403s.
        exc = RuntimeError("403 Forbidden")
        assert is_asknews_subscription_error(exc) is False

    def test_permission_error_does_not_match(self) -> None:
        # Built-in PermissionError has "forbidden path" wording but isn't the
        # SDK class name.
        exc = PermissionError("forbidden path /tmp")
        assert is_asknews_subscription_error(exc) is False

    def test_asyncio_timeout_error_does_not_match(self) -> None:
        # Timeout is an operational failure we DO want to alert on, not silence.
        exc = TimeoutError()
        assert is_asknews_subscription_error(exc) is False


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_FakeForbiddenError("403011 - subscription is not currently active"), True),
        (_FakeForbiddenError("some other error"), False),
        (RuntimeError("403 Forbidden"), False),
        (PermissionError("forbidden path /tmp"), False),
        (TimeoutError(), False),
    ],
)
def test_is_asknews_subscription_error_parametrized(exc: BaseException, expected: bool) -> None:
    """Parametrized smoke test mirroring the class-based tests above."""
    assert is_asknews_subscription_error(exc) is expected


class TestNimbleProvider:
    @pytest.mark.asyncio
    async def test_nimble_is_selected_first_and_formats_citations(self, monkeypatch):
        question = SimpleNamespace(
            question_text="Will the agency publish its annual estimate?",
            resolution_criteria="Resolves based on the official release.",
            fine_print="",
            id_of_question=42,
        )
        monkeypatch.setenv("NIMBLE_API_KEY", "test-key")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "asknews-id")
        monkeypatch.setenv("ASKNEWS_SECRET", "asknews-secret")
        provider, name = choose_provider_with_name()
        assert name == "nimble"

        request = {}

        async def fake_search(query, api_key, *, end_date=None):
            request["query"] = query
            request["api_key"] = api_key
            return {
                "results": [
                    {
                        "title": "Official update",
                        "url": "https://example.com/update",
                        "description": "The agency published a new figure.",
                    }
                ]
            }

        monkeypatch.setattr(providers, "_nimble_search", fake_search)
        research = await provider(question)

        assert request["api_key"] == "test-key"
        assert question.question_text in request["query"]
        assert "[Official update](https://example.com/update)" in research
        assert "The agency published a new figure." in research

    @pytest.mark.asyncio
    async def test_benchmark_search_is_bounded_to_resolution_date(self, monkeypatch):
        from datetime import datetime, timezone

        monkeypatch.setenv("NIMBLE_API_KEY", "test-key")
        question = SimpleNamespace(
            question_text="Will the agency publish its annual estimate?",
            resolution_criteria="",
            fine_print="",
            id_of_question=42,
            scheduled_resolution_time=datetime(2024, 10, 1, tzinfo=timezone.utc),
        )
        provider = providers._nimble_provider(is_benchmarking=True)
        request = {}

        async def fake_search(query, api_key, *, end_date=None):
            request["end_date"] = end_date
            return {"results": []}

        monkeypatch.setattr(providers, "_nimble_search", fake_search)
        assert await provider(question) == ""
        assert request["end_date"] == "2024-10-01"

    def test_forced_nimble_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("NIMBLE_API_KEY", raising=False)
        monkeypatch.setenv("RESEARCH_PROVIDER", "nimble")
        with pytest.raises(ValueError, match="NIMBLE_API_KEY"):
            choose_provider_with_name()
