"""Pattern inference and deterministic mixed-pattern assignment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from salvi.application.context import RunContext
from salvi.domain.enums import EvaluationIssueCode, PatternKind, PatternScope
from salvi.domain.models import (
    Bicluster,
    ColumnPatternFit,
    EvaluationIssue,
    PatternCandidateFit,
    PatternFit,
    PatternGroupFit,
)
from salvi.patterns.catalog import PatternCatalog, default_pattern_catalog
from salvi.patterns.contracts import (
    BatchColumnPatternFitter,
    ColumnPatternFitter,
    GroupPatternProposal,
    PatternImplementation,
)
from salvi.patterns.fitters.constant import ConstantPatternFitter
from salvi.patterns.math import NUMERIC_TOLERANCE, clamp01, diagnostics


def _with_reference_role(fit: PatternCandidateFit) -> PatternCandidateFit:
    return fit.model_copy(
        update={"diagnostics": tuple(sorted((*fit.diagnostics, ("reference_only", True))))}
    )


def _selection_diagnostics(
    selected: PatternCandidateFit,
    alternatives: tuple[PatternCandidateFit, ...],
) -> tuple[tuple[str, float | int | str | bool | None], ...]:
    """Expose observationally equivalent models without changing their fit."""

    assignable = tuple(
        alternative
        for alternative in alternatives
        if alternative.valid and not bool(dict(alternative.diagnostics).get("reference_only"))
    )
    competitors = tuple(
        alternative for alternative in assignable if alternative.pattern is not selected.pattern
    )
    if not competitors:
        return selected.diagnostics
    best_competing_error = min(alternative.error for alternative in competitors)
    gap = best_competing_error - selected.error
    equivalent = tuple(
        sorted(
            {
                selected.pattern.value,
                *(
                    alternative.pattern.value
                    for alternative in competitors
                    if abs(alternative.error - selected.error) <= NUMERIC_TOLERANCE
                ),
            }
        )
    )
    return tuple(
        sorted(
            (
                *selected.diagnostics,
                ("model_ambiguous", len(equivalent) > 1),
                ("model_error_margin", gap),
                ("model_equivalents", ",".join(equivalent)),
            )
        )
    )


@dataclass(frozen=True, slots=True)
class IterativeBestFitAssignment:
    """Assign local and joint patterns without assuming exactly two kinds.

    Local patterns establish independent alternatives. A single subset pattern is
    conservatively pruned one column at a time until its members improve over their
    local or null-model reference. Multiple one-group subset patterns first compete
    per column under their cardinality requirements and are then refitted on
    disjoint groups, preventing one acceptable group from monopolizing columns for
    which another pattern is a better explanation.
    """

    catalog: PatternCatalog

    def assign(
        self,
        context: RunContext,
        bicluster: Bicluster,
        implementations: tuple[PatternImplementation, ...],
    ) -> PatternFit:
        if (
            len(implementations) == 1
            and implementations[0].definition.kind is PatternKind.CONSTANT
            and isinstance(implementations[0].column_fitter, ConstantPatternFitter)
        ):
            return self._assign_constant_only(
                context,
                bicluster,
                implementations[0].column_fitter,
            )
        columns = bicluster.column_indices
        local_implementations = tuple(
            implementation
            for implementation in implementations
            if implementation.definition.scope is PatternScope.COLUMN
        )
        group_implementations = tuple(
            implementation
            for implementation in implementations
            if implementation.definition.scope is PatternScope.SUBSET
        )

        alternatives: dict[int, dict[PatternKind, PatternCandidateFit]] = {
            column: {} for column in columns
        }
        assignments: dict[int, PatternCandidateFit] = {}
        assignment_groups: dict[int, str] = {}
        for implementation in local_implementations:
            eligible = tuple(
                column
                for column in columns
                if implementation.definition.supports(
                    context.dataset.column_metadata(column).kind
                )
            )
            assert implementation.column_fitter is not None
            fits = self._fit_columns(
                context,
                bicluster,
                implementation.column_fitter,
                eligible,
            )
            for column, fit in zip(eligible, fits, strict=True):
                alternatives[column][fit.pattern] = fit
        for column in columns:
            valid = tuple(fit for fit in alternatives[column].values() if fit.valid)
            if valid:
                assignments[column] = min(valid, key=lambda fit: (fit.error, fit.pattern.value))

        references = self._local_or_reference_fits(
            context,
            bicluster,
            alternatives,
            assignments,
            group_implementations,
        )
        group_failures: dict[int, list[EvaluationIssue]] = {column: [] for column in columns}
        competing_groups = len(group_implementations) > 1 and all(
            implementation.definition.maximum_groups == 1
            for implementation in group_implementations
        )
        if competing_groups:
            groups = self._assign_competing_groups(
                context,
                bicluster,
                group_implementations,
                references,
                alternatives,
                assignments,
                assignment_groups,
                group_failures,
                has_local_fallback=bool(local_implementations),
            )
        else:
            groups = self._assign_sequential_groups(
                context,
                bicluster,
                group_implementations,
                references,
                alternatives,
                assignments,
                assignment_groups,
                group_failures,
                has_local_fallback=bool(local_implementations),
            )

        issues: list[EvaluationIssue] = []
        fitted_columns: list[ColumnPatternFit] = []
        for column in columns:
            selected_fit = assignments.get(column)
            candidate_alternatives = tuple(
                alternatives[column][alternative_kind]
                for alternative_kind in sorted(alternatives[column], key=lambda item: item.value)
            )
            if selected_fit is None:
                supported = any(
                    implementation.definition.supports(context.dataset.column_metadata(column).kind)
                    for implementation in implementations
                )
                column_issues = [
                    EvaluationIssue(
                        code=alternative.issue_code,
                        message=str(
                            dict(alternative.diagnostics).get(
                                "reason", "pattern alternative is invalid"
                            )
                        ),
                        column_index=column,
                        pattern=alternative.pattern,
                    )
                    for alternative in candidate_alternatives
                    if not alternative.valid and alternative.issue_code is not None
                ]
                column_issues.extend(group_failures[column])
                if not column_issues:
                    column_issues.append(
                        EvaluationIssue(
                            code=(
                                EvaluationIssueCode.PATTERN_UNASSIGNED
                                if supported
                                else EvaluationIssueCode.UNSUPPORTED_COLUMN_KIND
                            ),
                            message=("no allowed pattern produced a valid fit for this column"),
                            column_index=column,
                        )
                    )
                issues.extend(
                    sorted(
                        set(column_issues),
                        key=lambda issue: (
                            issue.pattern.value if issue.pattern is not None else "",
                            issue.code.value,
                        ),
                    )
                )
                fitted_columns.append(
                    ColumnPatternFit.model_construct(
                        column_index=column,
                        pattern=None,
                        error=1.0,
                        alternatives=candidate_alternatives,
                        diagnostics=diagnostics(reason="unassigned"),
                    )
                )
                continue
            fitted_columns.append(
                ColumnPatternFit.model_construct(
                    column_index=column,
                    pattern=selected_fit.pattern,
                    group_identifier=assignment_groups.get(column),
                    error=selected_fit.error,
                    parameter=selected_fit.parameter,
                    parameter_scale=selected_fit.parameter_scale,
                    source_support=selected_fit.source_support,
                    available_support=selected_fit.available_support,
                    prototype_support=selected_fit.prototype_support,
                    alternatives=candidate_alternatives,
                    diagnostics=_selection_diagnostics(
                        selected_fit,
                        candidate_alternatives,
                    ),
                )
            )
        return PatternFit.model_construct(
            candidate_signature=bicluster.signature,
            row_indices=bicluster.row_indices,
            column_indices=columns,
            columns=tuple(fitted_columns),
            groups=tuple(groups),
            issues=tuple(issues),
        )

    @staticmethod
    def _assign_constant_only(
        context: RunContext,
        bicluster: Bicluster,
        fitter: ConstantPatternFitter,
    ) -> PatternFit:
        alternatives = fitter.fit_columns(context, bicluster)
        columns: list[ColumnPatternFit] = []
        issues: list[EvaluationIssue] = []
        for column_index, alternative in zip(
            bicluster.column_indices,
            alternatives,
            strict=True,
        ):
            if not alternative.valid:
                assert alternative.issue_code is not None
                reason = str(
                    dict(alternative.diagnostics).get(
                        "reason",
                        "pattern alternative is invalid",
                    )
                )
                issues.append(
                    EvaluationIssue(
                        code=alternative.issue_code,
                        message=reason,
                        column_index=column_index,
                        pattern=PatternKind.CONSTANT,
                    )
                )
                columns.append(
                    ColumnPatternFit.model_construct(
                        column_index=column_index,
                        pattern=None,
                        error=1.0,
                        alternatives=(alternative,),
                        diagnostics=diagnostics(reason="unassigned"),
                    )
                )
                continue
            columns.append(
                ColumnPatternFit.model_construct(
                    column_index=column_index,
                    pattern=alternative.pattern,
                    error=alternative.error,
                    parameter=alternative.parameter,
                    parameter_scale=alternative.parameter_scale,
                    source_support=alternative.source_support,
                    available_support=alternative.available_support,
                    prototype_support=alternative.prototype_support,
                    alternatives=(alternative,),
                    diagnostics=alternative.diagnostics,
                )
            )
        return PatternFit.model_construct(
            candidate_signature=bicluster.signature,
            row_indices=bicluster.row_indices,
            column_indices=bicluster.column_indices,
            columns=tuple(columns),
            issues=tuple(issues),
        )

    def _local_or_reference_fits(
        self,
        context: RunContext,
        bicluster: Bicluster,
        alternatives: dict[int, dict[PatternKind, PatternCandidateFit]],
        assignments: dict[int, PatternCandidateFit],
        group_implementations: tuple[PatternImplementation, ...],
    ) -> dict[int, PatternCandidateFit]:
        reference_implementation = self.catalog.reference_implementation
        reference_kind = reference_implementation.definition.kind
        assert reference_implementation.column_fitter is not None
        references: dict[int, PatternCandidateFit] = {}
        missing: list[int] = []
        reference_columns = tuple(
            column
            for column in bicluster.column_indices
            if any(
                implementation.definition.supports(
                    context.dataset.column_metadata(column).kind
                )
                for implementation in group_implementations
            )
        )
        for column in reference_columns:
            local = assignments.get(column)
            if local is not None:
                references[column] = local
                continue
            existing = alternatives[column].get(reference_kind)
            if existing is not None:
                references[column] = existing
                continue
            kind = context.dataset.column_metadata(column).kind
            if reference_implementation.definition.supports(kind):
                missing.append(column)
        fitted_references = self._fit_columns(
            context,
            bicluster,
            reference_implementation.column_fitter,
            tuple(missing),
        )
        for column, fit in zip(missing, fitted_references, strict=True):
            reference = _with_reference_role(fit)
            alternatives[column][reference_kind] = reference
            references[column] = reference
        return references

    @staticmethod
    def _fit_columns(
        context: RunContext,
        bicluster: Bicluster,
        fitter: ColumnPatternFitter,
        columns: tuple[int, ...],
    ) -> tuple[PatternCandidateFit, ...]:
        if not columns:
            return ()
        if isinstance(fitter, BatchColumnPatternFitter):
            fits = tuple(fitter.fit_columns(context, bicluster, columns))
        else:
            fits = tuple(
                fitter.fit_column(context, bicluster, column) for column in columns
            )
        if len(fits) != len(columns):
            raise RuntimeError("batch column fitter returned an unexpected number of fits")
        return fits

    @classmethod
    def _assign_sequential_groups(
        cls,
        context: RunContext,
        bicluster: Bicluster,
        implementations: tuple[PatternImplementation, ...],
        references: dict[int, PatternCandidateFit],
        alternatives: dict[int, dict[PatternKind, PatternCandidateFit]],
        assignments: dict[int, PatternCandidateFit],
        assignment_groups: dict[int, str],
        group_failures: dict[int, list[EvaluationIssue]],
        *,
        has_local_fallback: bool,
    ) -> tuple[PatternGroupFit, ...]:
        """Retain the established greedy strategy for non-competing group models."""

        claimed: set[int] = set()
        groups: list[PatternGroupFit] = []
        group_counts: dict[PatternKind, int] = {}
        while implementations:
            proposals: list[tuple[float, GroupPatternProposal, PatternImplementation]] = []
            for implementation in implementations:
                maximum = implementation.definition.maximum_groups
                if (
                    maximum is not None
                    and group_counts.get(implementation.definition.kind, 0) >= maximum
                ):
                    continue
                eligible = tuple(
                    column
                    for column in bicluster.column_indices
                    if column not in claimed
                    and implementation.definition.supports(
                        context.dataset.column_metadata(column).kind
                    )
                    and references.get(column) is not None
                    and references[column].valid
                )
                if (
                    implementation.group_candidate_generator is None
                    or not has_local_fallback
                ):
                    proposal, proposal_issues = cls._stable_group_proposal(
                        context,
                        bicluster,
                        implementation,
                        eligible,
                        references,
                        has_local_fallback=has_local_fallback,
                    )
                else:
                    proposal, proposal_issues = cls._best_discovered_group_proposal(
                        context,
                        bicluster,
                        implementation,
                        eligible,
                        references,
                        has_local_fallback=has_local_fallback,
                        require_evidence=True,
                    )
                cls._record_group_failures(group_failures, proposal_issues)
                if proposal is None:
                    continue
                for column, candidate_fit in proposal.columns:
                    alternatives[column][candidate_fit.pattern] = candidate_fit
                improvements = tuple(
                    references[column].error - fit.error for column, fit in proposal.columns
                )
                proposals.append(
                    (
                        sum(improvements) / len(improvements),
                        proposal,
                        implementation,
                    )
                )
            if not proposals:
                break
            _, selected_proposal, implementation = min(
                proposals,
                key=lambda item: (-item[0], item[1].pattern.value),
            )
            assert selected_proposal.group is not None
            pattern = implementation.definition.kind
            group_number = group_counts.get(pattern, 0)
            group_identifier = f"{pattern.value}-{group_number}"
            cls._commit_group(
                context,
                selected_proposal,
                references,
                alternatives,
                assignments,
                assignment_groups,
                group_identifier,
                has_local_fallback=has_local_fallback,
            )
            claimed.update(column for column, _ in selected_proposal.columns)
            groups.append(
                selected_proposal.group.model_copy(update={"identifier": group_identifier})
            )
            group_counts[pattern] = group_number + 1
        return tuple(groups)

    @classmethod
    def _assign_competing_groups(
        cls,
        context: RunContext,
        bicluster: Bicluster,
        implementations: tuple[PatternImplementation, ...],
        references: dict[int, PatternCandidateFit],
        alternatives: dict[int, dict[PatternKind, PatternCandidateFit]],
        assignments: dict[int, PatternCandidateFit],
        assignment_groups: dict[int, str],
        group_failures: dict[int, list[EvaluationIssue]],
        *,
        has_local_fallback: bool,
    ) -> tuple[PatternGroupFit, ...]:
        """Partition overlapping one-group models before their final refit."""

        implementations_by_pattern = {
            implementation.definition.kind: implementation
            for implementation in implementations
        }
        proposal_fits: dict[PatternKind, dict[int, PatternCandidateFit]] = {}
        for implementation in implementations:
            eligible = tuple(
                column
                for column in bicluster.column_indices
                if implementation.definition.supports(
                    context.dataset.column_metadata(column).kind
                )
                and references.get(column) is not None
                and references[column].valid
            )
            proposal, proposal_issues = cls._best_discovered_group_proposal(
                context,
                bicluster,
                implementation,
                eligible,
                references,
                has_local_fallback=has_local_fallback,
                require_evidence=False,
            )
            cls._record_group_failures(group_failures, proposal_issues)
            if proposal is None:
                continue
            fits = dict(proposal.columns)
            proposal_fits[implementation.definition.kind] = fits
            for column, fit in proposal.columns:
                alternatives[column][fit.pattern] = fit

        active: set[PatternKind] = set()
        choices: dict[int, PatternKind] = {}
        best_key: tuple[float, int, int, tuple[str, ...]] | None = None
        available_patterns = tuple(sorted(proposal_fits, key=lambda pattern: pattern.value))
        for pattern_count in range(1, len(available_patterns) + 1):
            for selected_patterns in combinations(available_patterns, pattern_count):
                candidate_choices: dict[int, PatternKind] = {}
                score = 0.0
                for column, reference in references.items():
                    options = tuple(
                        (proposal_fits[pattern][column].error, pattern.value, pattern)
                        for pattern in selected_patterns
                        if column in proposal_fits[pattern]
                        and (
                            not has_local_fallback
                            or reference.error - proposal_fits[pattern][column].error
                            >= context.patterns.min_improvement
                        )
                    )
                    if not options:
                        continue
                    selected_pattern = min(options)[2]
                    candidate_choices[column] = selected_pattern
                    score += reference.error - proposal_fits[selected_pattern][column].error
                if any(
                    sum(choice is pattern for choice in candidate_choices.values())
                    < implementations_by_pattern[pattern].definition.minimum_columns
                    for pattern in selected_patterns
                ):
                    continue
                candidate_key = (
                    -score,
                    -len(candidate_choices),
                    len(selected_patterns),
                    tuple(pattern.value for pattern in selected_patterns),
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    active = set(selected_patterns)
                    choices = candidate_choices

        groups: list[PatternGroupFit] = []
        for implementation in implementations:
            pattern = implementation.definition.kind
            if pattern not in active:
                continue
            assigned = tuple(
                column
                for column in bicluster.column_indices
                if choices.get(column) is pattern
            )
            proposal, proposal_issues = cls._stable_group_proposal(
                context,
                bicluster,
                implementation,
                assigned,
                references,
                has_local_fallback=has_local_fallback,
            )
            cls._record_group_failures(group_failures, proposal_issues)
            if proposal is None:
                continue
            group_identifier = f"{pattern.value}-0"
            cls._commit_group(
                context,
                proposal,
                references,
                alternatives,
                assignments,
                assignment_groups,
                group_identifier,
                has_local_fallback=has_local_fallback,
            )
            assert proposal.group is not None
            groups.append(proposal.group.model_copy(update={"identifier": group_identifier}))
        return tuple(groups)

    @classmethod
    def _best_discovered_group_proposal(
        cls,
        context: RunContext,
        bicluster: Bicluster,
        implementation: PatternImplementation,
        eligible: tuple[int, ...],
        references: dict[int, PatternCandidateFit],
        *,
        has_local_fallback: bool,
        require_evidence: bool,
    ) -> tuple[GroupPatternProposal | None, tuple[EvaluationIssue, ...]]:
        generator = implementation.group_candidate_generator
        if generator is None:
            return cls._stable_group_proposal(
                context,
                bicluster,
                implementation,
                eligible,
                references,
                has_local_fallback=has_local_fallback,
            )
        candidates = tuple(
            dict.fromkeys(
                generator.propose(
                    context,
                    bicluster,
                    eligible,
                    minimum_columns=implementation.definition.minimum_columns,
                )
            )
        )
        if not candidates:
            return cls._stable_group_proposal(
                context,
                bicluster,
                implementation,
                eligible,
                references,
                has_local_fallback=has_local_fallback,
            )
        proposals: list[tuple[tuple[float, int, float, tuple[int, ...]], GroupPatternProposal]] = []
        issues: list[EvaluationIssue] = []
        for candidate_columns in candidates:
            if require_evidence:
                proposal, proposal_issues = cls._stable_group_proposal(
                    context,
                    bicluster,
                    implementation,
                    candidate_columns,
                    references,
                    has_local_fallback=has_local_fallback,
                )
            else:
                proposal, proposal_issues = cls._supported_group_proposal(
                    context,
                    bicluster,
                    implementation,
                    candidate_columns,
                )
            issues.extend(proposal_issues)
            if proposal is None:
                continue
            improvements = tuple(
                references[column].error - fit.error for column, fit in proposal.columns
            )
            key = (
                -sum(improvements),
                -len(proposal.columns),
                sum(fit.error for _, fit in proposal.columns) / len(proposal.columns),
                tuple(column for column, _ in proposal.columns),
            )
            proposals.append((key, proposal))
        if not proposals:
            return None, tuple(issues)
        return min(proposals, key=lambda item: item[0])[1], tuple(issues)

    @staticmethod
    def _supported_group_proposal(
        context: RunContext,
        bicluster: Bicluster,
        implementation: PatternImplementation,
        eligible: tuple[int, ...],
    ) -> tuple[GroupPatternProposal | None, tuple[EvaluationIssue, ...]]:
        """Remove only columns that cannot mathematically participate in a proposed group."""

        assert implementation.group_fitter is not None
        candidates = list(eligible)
        minimum = implementation.definition.minimum_columns
        issues: list[EvaluationIssue] = []
        while len(candidates) >= minimum:
            proposal = implementation.group_fitter.fit_group(
                context,
                bicluster,
                tuple(candidates),
            )
            if proposal.rejected_column is not None:
                assert proposal.rejection_issue is not None
                issues.append(proposal.rejection_issue)
                candidates.remove(proposal.rejected_column)
                continue
            return (proposal if proposal.valid else None), tuple(issues)
        return None, tuple(issues)

    @staticmethod
    def _record_group_failures(
        failures: dict[int, list[EvaluationIssue]],
        issues: tuple[EvaluationIssue, ...],
    ) -> None:
        for issue in issues:
            assert issue.column_index is not None
            failures[issue.column_index].append(issue)

    @staticmethod
    def _commit_group(
        context: RunContext,
        proposal: GroupPatternProposal,
        references: dict[int, PatternCandidateFit],
        alternatives: dict[int, dict[PatternKind, PatternCandidateFit]],
        assignments: dict[int, PatternCandidateFit],
        assignment_groups: dict[int, str],
        group_identifier: str,
        *,
        has_local_fallback: bool,
    ) -> None:
        for column, candidate_fit in proposal.columns:
            final_fit = candidate_fit
            if not has_local_fallback:
                threshold = context.patterns.min_improvement
                improvement = references[column].error - candidate_fit.error
                deficit = (
                    0.0 if threshold == 0.0 else clamp01((threshold - improvement) / threshold)
                )
                final_fit = candidate_fit.model_copy(
                    update={
                        "error": max(candidate_fit.error, deficit),
                        "diagnostics": tuple(
                            sorted(
                                (
                                    *candidate_fit.diagnostics,
                                    ("evidence_deficit", deficit),
                                )
                            )
                        ),
                    }
                )
            alternatives[column][final_fit.pattern] = final_fit
            assignments[column] = final_fit
            assignment_groups[column] = group_identifier

    @staticmethod
    def _stable_group_proposal(
        context: RunContext,
        bicluster: Bicluster,
        implementation: PatternImplementation,
        eligible: tuple[int, ...],
        references: dict[int, PatternCandidateFit],
        *,
        has_local_fallback: bool,
    ) -> tuple[GroupPatternProposal | None, tuple[EvaluationIssue, ...]]:
        assert implementation.group_fitter is not None
        candidates = list(eligible)
        minimum = implementation.definition.minimum_columns
        issues: list[EvaluationIssue] = []
        while len(candidates) >= minimum:
            proposal = implementation.group_fitter.fit_group(context, bicluster, tuple(candidates))
            if proposal.rejected_column is not None:
                assert proposal.rejection_issue is not None
                issues.append(proposal.rejection_issue)
                candidates.remove(proposal.rejected_column)
                continue
            if not proposal.valid:
                return None, tuple(issues)
            if not has_local_fallback:
                return proposal, tuple(issues)
            deficits = tuple(
                (
                    context.patterns.min_improvement - (references[column].error - fit.error),
                    column,
                )
                for column, fit in proposal.columns
            )
            worst_deficit, rejected = max(deficits, key=lambda item: (item[0], -item[1]))
            if worst_deficit <= 0.0:
                return proposal, tuple(issues)
            candidates.remove(rejected)
        if not has_local_fallback:
            issues.extend(
                EvaluationIssue(
                    code=EvaluationIssueCode.INSUFFICIENT_GROUP_SUPPORT,
                    message=(
                        f"{implementation.definition.kind.value} requires at least "
                        f"{minimum} jointly fitted columns"
                    ),
                    column_index=column,
                    pattern=implementation.definition.kind,
                )
                for column in candidates
            )
        return None, tuple(issues)


@dataclass(frozen=True, slots=True)
class DefaultPatternInferenceEngine:
    catalog: PatternCatalog
    assignment: IterativeBestFitAssignment

    @classmethod
    def create(cls, catalog: PatternCatalog | None = None) -> DefaultPatternInferenceEngine:
        active_catalog = catalog or default_pattern_catalog()
        return cls(active_catalog, IterativeBestFitAssignment(active_catalog))

    def infer(self, context: RunContext, bicluster: Bicluster) -> PatternFit:
        implementations = self.catalog.implementations(context.patterns.allowed)
        return self.assignment.assign(context, bicluster, implementations)


__all__ = ["DefaultPatternInferenceEngine", "IterativeBestFitAssignment"]
