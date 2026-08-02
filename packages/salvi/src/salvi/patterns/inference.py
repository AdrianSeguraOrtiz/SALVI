"""Pattern inference and deterministic mixed-pattern assignment."""

from __future__ import annotations

from dataclasses import dataclass

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
    GroupPatternProposal,
    PatternImplementation,
)
from salvi.patterns.fitters.constant import ConstantPatternFitter
from salvi.patterns.math import clamp01, diagnostics


def _with_reference_role(fit: PatternCandidateFit) -> PatternCandidateFit:
    return fit.model_copy(
        update={"diagnostics": tuple(sorted((*fit.diagnostics, ("reference_only", True))))}
    )


@dataclass(frozen=True, slots=True)
class IterativeBestFitAssignment:
    """Assign local and joint patterns without assuming exactly two kinds.

    Local patterns establish independent alternatives. Every subset pattern is
    fitted on compatible, unclaimed columns and pruned one column at a time until
    its members improve over their best local or null-model reference. Among
    overlapping stable proposals, the largest mean improvement wins; remaining
    pattern implementations can then compete for the unclaimed columns.
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
        for column in columns:
            kind = context.dataset.column_metadata(column).kind
            for implementation in local_implementations:
                if not implementation.definition.supports(kind):
                    continue
                assert implementation.column_fitter is not None
                fit = implementation.column_fitter.fit_column(context, bicluster, column)
                alternatives[column][fit.pattern] = fit
            valid = tuple(fit for fit in alternatives[column].values() if fit.valid)
            if valid:
                assignments[column] = min(valid, key=lambda fit: (fit.error, fit.pattern.value))

        references = self._local_or_reference_fits(
            context,
            bicluster,
            alternatives,
            assignments,
        )
        claimed: set[int] = set()
        groups: list[PatternGroupFit] = []
        group_counts: dict[PatternKind, int] = {}
        group_failures: dict[int, list[EvaluationIssue]] = {column: [] for column in columns}
        while group_implementations:
            proposals: list[tuple[float, GroupPatternProposal, PatternImplementation]] = []
            for implementation in group_implementations:
                maximum = implementation.definition.maximum_groups
                if (
                    maximum is not None
                    and group_counts.get(implementation.definition.kind, 0) >= maximum
                ):
                    continue
                eligible = tuple(
                    column
                    for column in columns
                    if column not in claimed
                    and implementation.definition.supports(
                        context.dataset.column_metadata(column).kind
                    )
                    and references.get(column) is not None
                    and references[column].valid
                )
                proposal, proposal_issues = self._stable_group_proposal(
                    context,
                    bicluster,
                    implementation,
                    eligible,
                    references,
                    has_local_fallback=bool(local_implementations),
                )
                for issue in proposal_issues:
                    assert issue.column_index is not None
                    group_failures[issue.column_index].append(issue)
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
            _, selected_proposal, implementation = sorted(
                proposals,
                key=lambda item: (-item[0], item[1].pattern.value),
            )[0]
            assert selected_proposal.group is not None
            group_only = not local_implementations
            pattern_kind = implementation.definition.kind
            group_number = group_counts.get(pattern_kind, 0)
            group_identifier = f"{pattern_kind.value}-{group_number}"
            for column, candidate_fit in selected_proposal.columns:
                final_fit = candidate_fit
                reference = references[column]
                if group_only:
                    threshold = context.patterns.min_improvement
                    improvement = reference.error - candidate_fit.error
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
                claimed.add(column)
            groups.append(
                selected_proposal.group.model_copy(update={"identifier": group_identifier})
            )
            group_counts[pattern_kind] = group_counts.get(pattern_kind, 0) + 1

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
                    diagnostics=selected_fit.diagnostics,
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
    ) -> dict[int, PatternCandidateFit]:
        reference_implementation = self.catalog.reference_implementation
        reference_kind = reference_implementation.definition.kind
        assert reference_implementation.column_fitter is not None
        references: dict[int, PatternCandidateFit] = {}
        for column in bicluster.column_indices:
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
                reference = _with_reference_role(
                    reference_implementation.column_fitter.fit_column(context, bicluster, column)
                )
                alternatives[column][reference_kind] = reference
                references[column] = reference
        return references

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
