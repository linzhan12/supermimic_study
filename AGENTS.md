# Repository workflow

Before making code changes, create and switch to a new descriptive Git branch from the current starting point (for example, `codex/YYYYMMDD-short-description`). If the worktree has existing changes, preserve them and do not overwrite or discard them. Do not switch branches when doing so would risk unrelated work; explain any blocker instead.

Commit completed, verified work when the user asks for a commit. Keep commits focused so a change can be reverted with `git revert <commit>`; branches alone do not preserve uncommitted changes. Never reset or discard user changes without explicit permission.
