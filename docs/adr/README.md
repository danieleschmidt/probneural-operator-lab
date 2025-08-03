# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records for the ProbNeural-Operator-Lab project.

## ADR Format

We use the format described by Michael Nygard in [Documenting Architecture Decisions](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions).

Each ADR should follow this structure:

```markdown
# ADR-XXXX: [Title]

## Status
[Proposed | Accepted | Rejected | Deprecated | Superseded by ADR-YYYY]

## Context
What is the issue that we're seeing that is motivating this decision or change?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult to do because of this change?
```

## ADR Index

- [ADR-0001: Use Linearized Laplace for Primary Uncertainty Method](0001-linearized-laplace-primary.md)
- [ADR-0002: Modular Architecture for Neural Operators](0002-modular-architecture.md)
- [ADR-0003: PyTorch as Primary Backend](0003-pytorch-backend.md)
- [ADR-0004: Hierarchical Package Structure](0004-hierarchical-package-structure.md)

## Creating New ADRs

1. Create a new file: `docs/adr/NNNN-short-title.md`
2. Use the next sequential number (NNNN)
3. Follow the template format above
4. Update this index when adding new ADRs