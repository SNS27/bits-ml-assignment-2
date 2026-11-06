#!/usr/bin/env bash
# Read-only environment + Git check. Paste the output to Claude if anything looks wrong.
echo "== where =="; pwd; uname -a 2>/dev/null; echo "shell: $SHELL"
echo "== tools =="; (python --version || python3 --version) 2>&1; (pip --version || python3 -m pip --version) 2>&1; git --version
echo "== git =="; git rev-parse --show-toplevel 2>&1; git branch --show-current 2>&1; git status -sb 2>&1 | head -15
git remote -v 2>&1; git log --oneline -5 2>&1
echo "== remote reachability =="
git ls-remote https://github.com/SNS27/bits-ml-assignment-2.git
rc=$?
if [ $rc -eq 0 ]; then echo "remote reachable (empty output above = empty repo; lines = existing commits)";
else echo "ls-remote failed (exit $rc): authentication, wrong repo name, or network problem, NOT an empty repo"; fi
echo "== identity =="; git config user.name; git config user.email
