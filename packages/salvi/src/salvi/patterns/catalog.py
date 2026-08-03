"""Deterministic catalog of available pattern implementations."""

from __future__ import annotations

from collections.abc import Iterable

from salvi.domain.enums import PatternKind
from salvi.patterns.contracts import PatternDefinition, PatternImplementation


class PatternCatalog:
    def __init__(self, implementations: Iterable[PatternImplementation]) -> None:
        ordered = tuple(sorted(implementations, key=lambda item: item.definition.kind.value))
        kinds = tuple(item.definition.kind for item in ordered)
        if not ordered:
            raise ValueError("a pattern catalog cannot be empty")
        if len(set(kinds)) != len(kinds):
            raise ValueError("pattern implementations must have unique kinds")
        references = tuple(item for item in ordered if item.definition.reference_model)
        if len(references) != 1:
            raise ValueError("a pattern catalog requires exactly one reference model")
        self._implementations = ordered
        self._by_kind = {item.definition.kind: item for item in ordered}
        self._reference = references[0]

    def implementation(self, kind: PatternKind) -> PatternImplementation:
        try:
            return self._by_kind[kind]
        except KeyError as error:
            raise ValueError(f"pattern {kind.value} is not registered") from error

    def implementations(
        self, allowed: Iterable[PatternKind] | None = None
    ) -> tuple[PatternImplementation, ...]:
        if allowed is None:
            return self._implementations
        selected = frozenset(allowed)
        unknown = selected - self._by_kind.keys()
        if unknown:
            names = ", ".join(sorted(kind.value for kind in unknown))
            raise ValueError(f"unregistered allowed patterns: {names}")
        return tuple(
            implementation
            for implementation in self._implementations
            if implementation.definition.kind in selected
        )

    def definitions(self) -> tuple[PatternDefinition, ...]:
        return tuple(item.definition for item in self._implementations)

    @property
    def reference_implementation(self) -> PatternImplementation:
        return self._reference


def default_pattern_catalog(
    allowed: Iterable[PatternKind] | None = None,
) -> PatternCatalog:
    from salvi.patterns.contrast import (
        AdditivePatternContrastStrategy,
        ConstantPatternContrastStrategy,
        MultiplicativePatternContrastStrategy,
    )
    from salvi.patterns.fitters import (
        AdditivePatternFitter,
        ConstantPatternFitter,
        MultiplicativePatternFitter,
    )
    from salvi.patterns.group_discovery import (
        AdditiveNeighborhoodGenerator,
        MultiplicativeNeighborhoodGenerator,
    )
    from salvi.patterns.seeding import (
        AdditivePatternSeedStrategy,
        ConstantPatternSeedStrategy,
        MultiplicativePatternSeedStrategy,
    )

    selected = frozenset(PatternKind) if allowed is None else frozenset(allowed)
    constant = ConstantPatternFitter()
    implementations = [
        PatternImplementation(
            definition=constant.definition,
            contrast_strategy=ConstantPatternContrastStrategy(),
            column_fitter=constant,
            seed_strategy=ConstantPatternSeedStrategy(constant.definition),
        )
    ]
    if PatternKind.ADDITIVE in selected:
        additive = AdditivePatternFitter()
        implementations.append(
            PatternImplementation(
                definition=additive.definition,
                contrast_strategy=AdditivePatternContrastStrategy(),
                group_fitter=additive,
                group_candidate_generator=AdditiveNeighborhoodGenerator(),
                seed_strategy=AdditivePatternSeedStrategy(additive.definition),
            )
        )
    if PatternKind.MULTIPLICATIVE in selected:
        multiplicative = MultiplicativePatternFitter()
        implementations.append(
            PatternImplementation(
                definition=multiplicative.definition,
                contrast_strategy=MultiplicativePatternContrastStrategy(),
                group_fitter=multiplicative,
                group_candidate_generator=MultiplicativeNeighborhoodGenerator(),
                seed_strategy=MultiplicativePatternSeedStrategy(multiplicative.definition),
            )
        )
    return PatternCatalog(implementations)


__all__ = ["PatternCatalog", "default_pattern_catalog"]
