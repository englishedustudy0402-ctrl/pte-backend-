"""Route-registration and security regression guards for the pipeline.

These lock in the fixes made during the audit without touching the live stack:
  - /questions/{question_id} is registered AFTER /next, /random, /by-key,
    /stats/me etc. so the catch-all never shadows them (STEP 2 bug).
  - /attempts/{attempt_id} never shadows /attempts/my (same class of bug).
  - sensitive endpoints attach auth dependencies.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("SUPABASE_URL", "http://127.0.0.1:54321")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "placeholder")

import main  # noqa: E402


def _routes(prefix):
    out = []
    for r in main.app.routes:
        path = getattr(r, "path", "")
        if path.startswith(prefix):
            for method in getattr(r, "methods", None) or []:
                out.append((method, path))
    return out


def _path_index(prefix, needle):
    order = [path for (_, path) in _routes(prefix) if path != prefix.rstrip("/")]
    return order.index(needle)


class TestQuestionRoutes:
    def test_dynamic_is_last(self):
        order = [path for (_, path) in _routes("/questions") if not path.endswith("/")]
        other = [p for p in order if p != "/questions/{question_id}"]
        assert all(order.index("/questions/{question_id}") > order.index(p) for p in other)

    def test_subresource_shape(self):
        paths = {p for (_, p) in _routes("/questions")}
        assert {"/questions/", "/questions/next", "/questions/random",
                "/questions/stats/me", "/questions/{question_id}"} <= paths


class TestAttemptRoutes:
    def test_my_before_dynamic(self):
        assert _path_index("/attempts", "/attempts/my") < _path_index("/attempts", "/attempts/{attempt_id}")

    def test_endpoints_present(self):
        pairs = _routes("/attempts")
        for path in ("/attempts/", "/attempts/my", "/attempts/{attempt_id}",
                     "/attempts/{attempt_id}/submit", "/attempts/{attempt_id}/feedback"):
            assert any(p == path for (_, p) in pairs), path


class TestAnalyticsRoutes:
    def test_endpoints_present(self):
        paths = {p for (_, p) in _routes("/analytics")}
        assert {"/analytics/overview", "/analytics/skills", "/analytics/trend"} <= paths


class TestAuthGating:
    def test_attempt_create_requires_plan(self):
        from fastapi.routing import APIRoute
        routes = [r for r in main.app.routes
                  if isinstance(r, APIRoute) and r.path == "/attempts/" and "POST" in r.methods]
        assert routes, "POST /attempts/ route must exist"
        names = [d.call.__qualname__ for d in routes[0].dependant.dependencies]
        assert any("require_active_plan" in n for n in names), names


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))