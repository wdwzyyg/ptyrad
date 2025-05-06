# 🌀 Git Workflow for PtyRAD

This is a lightweight Git workflow that combines aspects of Git Flow and GitHub Flow. The key distinction is the use of **temporary `dev-<DATE>` branches** to accumulate features before cleanly integrating them into `main`.

## 🌲 Branch Structure

| Branch           | Purpose                                                                      |
|------------------|------------------------------------------------------------------------------|
| `main`           | Permanent. Holds clean, linear release history.                              |
| `dev-<DATE>`     | Temporary branch per dev cycle. Integrates features before merging into main |
| `feature/*`      | Short-lived branches for individual features, branched from `dev-<DATE>`.    |


## ✅ Guidelines
- There should be only one active `dev-<DATE>` branch at a time.
- Each commit on `main` corresponds to a completed feature.
- Rebasing before merging avoids messy merge commits.
- Use squash merges for features, and rebase merges from `dev-<DATE>` to `main`.

## :arrows_counterclockwise: Development Workflow
1. **Start a development cycle**  
   Create a temporary development branch from `main`, named `dev-<DATE>`, e.g., `dev-20250419`.  
   This will be the only active dev branch during the cycle.

2. **Create a feature branch**  
   Branch off `dev-<DATE>` with a descriptive name, like `feature/<name>`.  
   Make sure your `dev-<DATE>` is up to date before branching, or just branch off from the remote `dev-<DATE>`.

3. **Develop the feature**  
   Add commits on your `feature/*` branch as usual. Use clear, meaningful messages.  
   Keep your branch focused and try to avoid bundling unrelated changes.

4. **Keep your feature branch up to date**  
   If other features have been merged into `dev-<DATE>` while you were developing,  
   **rebase your branch onto the latest `dev-<DATE>`** before opening a PR:  
   - This reduces merge conflicts later  
   - Be sure to pull the latest `dev-<DATE>` before rebasing  
   - Use `--force-with-lease` when pushing after rebasing

5. **Open a pull request (PR)**  
   Create a PR from your `feature/*` branch into `dev-<DATE>`.  
   Confirm it includes only the intended changes.

6. **Squash-merge the PR into `dev-<DATE>`**  
   After approval, squash-merge the PR to combine your commits into a single feature commit.  
   Delete the `feature/*` branch after merge.

7. **Repeat steps 2–6**  
   Continue developing features this way until `dev-<DATE>` is ready for release.

8. **Prepare for release**  
   Once development is complete, rebase `dev-<DATE>` onto the latest `main` to pick up any important changes happened on `main` during development (i.e., hotfixes).  
   Resolve conflicts if needed.

9. **Rebase-merge the `dev-<DATE>` branch into the `main`**  
   Open a PR from `dev-<DATE>` into `main`, and use **rebase-merge** (not squash).  
   This will preserve the individual feature commits with a clean linear history.  
   After merge, delete the `dev-<DATE>` branch.

10. **Tag and bump the version**  
    Bump the version as part of the merge into `main`, and optionally tag the release.

## :fire: Hotfix workflow

> A hotfix is a patch that needs to go directly into `main` without going through the regular release process. 
The most common use case is to patch a bug that's on production when `dev-<DATE>` contains code that isn't yet ready for release.

1. **Create a hotfix branch from `main`** to patch a critical issue discovered in production.

2. **Open a pull request into `main`** and squash-merge the fix to maintain a clean release history. Optionally tag the patch release.

3. **Backport the hotfix to the active `dev-<DATE>` branch** to ensure the fix is included in the next release cycle. This is typically done via `cherry-pick` to bring over just the relevant commit(s).

4. **Resolve any conflicts manually** during the cherry-pick process and complete the integration into `dev-<DATE>`.

5. **If necessary, rebase or update any in-progress feature branches** that were branched from `dev-<DATE>` **before the hotfix was applied**, to ensure consistency and avoid conflicts later when merging into `dev-<DATE>`.

6. **Communicate clearly** with contributors working on feature branches so they can rebase against the updated `dev-<DATE>` and resolve conflicts early.