python := ".venv/bin/python"
repo := "EduBalbino/pqmflib-api"
pr_number := "1"

default:
    @just --list

# Install Lefthook first: https://lefthook.dev/installation/
hooks:
    lefthook install
    {{python}} scripts/heisenberg_workflow.py install

# Measure the current committed implementation before editing it.
reference:
    {{python}} scripts/heisenberg_workflow.py reference

# Measure the candidate and check exact expectations.
benchmark:
    {{python}} scripts/heisenberg_workflow.py benchmark

# Require at least 10% less runtime than the previous accepted implementation.
check:
    {{python}} scripts/heisenberg_workflow.py check

# Internal post-commit hook: record the accepted measurement.
accept:
    {{python}} scripts/heisenberg_workflow.py accept

# Validate, locally commit, push, and sync a passing improvement in one operation.
promote:
    {{python}} scripts/heisenberg_workflow.py publish {{repo}} {{pr_number}}

# Retry PR synchronization for an already accepted commit.
pr:
    {{python}} scripts/heisenberg_workflow.py pr {{repo}} {{pr_number}}
