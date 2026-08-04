"""Publication route gates. Automatic Graph publishing is opt-in and fail-closed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphGates:
    explicitly_enabled: bool = False
    capability_passed: bool = False
    auth_passed: bool = False
    transport_passed: bool = False
    reconciliation_passed: bool = False

    @property
    def ready(self) -> bool:
        return all(
            (
                self.explicitly_enabled,
                self.capability_passed,
                self.auth_passed,
                self.transport_passed,
                self.reconciliation_passed,
            )
        )

    @property
    def route(self) -> str:
        return "graph_api" if self.ready else "manual_ui_package"


DEFAULT_GRAPH_GATES = GraphGates()
