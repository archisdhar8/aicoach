from typing import Protocol

from app.schemas.domain import Play, SimulationFrame


class SimulationEngine(Protocol):
    def simulate(self, play: Play) -> list[SimulationFrame]: ...
