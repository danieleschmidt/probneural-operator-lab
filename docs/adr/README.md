# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for ProbNeural-Operator-Lab. ADRs are documents that capture important architectural decisions made during the development of the project, along with their context and consequences.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences. ADRs help teams understand:
- What decision was made
- Why it was made
- What alternatives were considered
- What the consequences are

## ADR Format

We use a lightweight ADR format with the following sections:

```markdown
# ADR-XXXX: [Decision Title]

**Date**: YYYY-MM-DD  
**Status**: [Proposed | Accepted | Superseded | Deprecated]  
**Deciders**: [List of people involved in the decision]

## Context
Brief description of the situation and problem that requires a decision.

## Decision
The decision that was made and why.

## Alternatives Considered
Other options that were evaluated.

## Consequences
What becomes easier or more difficult as a result of this decision.

## Implementation Notes
Technical details or implementation considerations (if applicable).
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](adr-0001-pytorch-backend.md) | PyTorch as Primary Backend | Accepted | 2025-08-02 |
| [ADR-0002](adr-0002-laplace-approximation.md) | Linearized Laplace as Primary Uncertainty Method | Accepted | 2025-08-02 |
| [ADR-0003](adr-0003-modular-architecture.md) | Modular Component Architecture | Accepted | 2025-08-02 |

## Creating New ADRs

When making significant architectural decisions:

1. **Copy the template**: Use the format above as a starting point
2. **Assign a number**: Use the next sequential ADR number
3. **Fill out all sections**: Provide comprehensive context and rationale
4. **Review with team**: Discuss with stakeholders before finalizing
5. **Update this index**: Add the new ADR to the table above

## ADR Lifecycle

- **Proposed**: Decision is under consideration
- **Accepted**: Decision has been made and is being implemented
- **Superseded**: Decision has been replaced by a newer ADR
- **Deprecated**: Decision is no longer relevant but kept for historical context

## Guidelines

### When to Write an ADR
- Architectural patterns or frameworks
- Technology choices (backends, dependencies)
- Design patterns and interfaces
- Performance or scalability decisions
- Security or compliance requirements

### When NOT to Write an ADR
- Implementation details that don't affect architecture
- Temporary workarounds or bug fixes
- Configuration changes
- Documentation updates

### Writing Tips
- Be concise but comprehensive
- Focus on the "why" not just the "what"
- Include relevant technical details
- Consider future maintenance and evolution
- Reference related ADRs or external documentation

## References

- [Michael Nygard's ADR process](http://thinkrelevance.com/blog/2011/11/15/documenting-architecture-decisions)
- [ADR Tools](https://github.com/npryce/adr-tools)
- [Architecture Decision Records Best Practices](https://engineering.atspotify.com/2020/04/14/when-should-i-write-an-architecture-decision-record/)