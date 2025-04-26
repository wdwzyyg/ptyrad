# 🧩 Contributing to PtyRAD

Thank you for your interest in contributing to **PtyRAD**!
This document explains how we collaborate, the Git workflow we follow, and how to get started with contributing features or fixes.


## 💬 Personal Note from the Maintainer

Hey there! I want to share a bit of context to help you understand how this project runs.

**PtyRAD** is still actively under development. While I’ve done my best to ship an early version to serve the community, the work is far from finished. I’m continuing to refine and expand the codebase as I conduct more ptychography research. For now, I’d like to keep a good amount of creative and technical control over how the project evolves. I hope you understand — this is just to ensure the direction stays coherent while the foundations are still forming.

I'll continue to maintain PtyRAD for as long as I'm still doing ptychography research. That said, my availability is somewhat tied to my research projects and my current career stage. As of 2025, I’m a postdoc actively looking for faculty positions, so development pace may undergo ups and downs depending on how things unfold.

The development cycle is mainly guided by my own research needs, but I also try to blend in requested features and community contributions wherever possible. If something overlaps with my work, I’ll likely prioritize it — but even when it doesn’t, I’m always happy to explore new ideas together.

Also, to be totally transparent: I’m not a professional software engineer, and collaborative coding is still a relatively new experience for me. I’ve mostly worked solo in the past, so setting up a GitHub project and opening the doors to contributions is both exciting and a learning process. Thanks for bearing with me as I figure out what works best.

The Git workflow I’m using is tailored to my current needs and preferences. It helps me focus on short-term goals while staying on top of changes without getting overwhelmed. I review every PR myself, and I want each development cycle to be intentional, readable, and easy to follow.

That’s why I’ve created a list of wanted features (see below). If you’re interested in contributing, just pick one that excites you and let me know. I’m happy to guide you through the process, and I really appreciate the support.

Thanks for being here and helping make PtyRAD better!

— Chia-Hao Lee, 2025.04.19

## 🚦 Development Workflow Overview

We follow a [**cycle-based Git workflow**](./docs/GIT_WORKFLOW.md) using temporary dev branches and squash/rebase merges:

### 🔁 General Flow
1. The permanent branch is `main`, containing only stable, tagged releases.
2. A temporary branch `dev-<DATE>` (e.g., `dev-20250419`) is created from `main` to represent the current development cycle.
3. Contributors create short-lived branches from `dev-<DATE>`:
   - `feature/*` for new features
   - `bugfix/*` for minor fixes
   - `docs/*` for documentation
4. Once a feature is complete, it is squash-merged into `dev-<DATE>` to keep history clean (1 feature = 1 commit).
5. When the cycle ends, `dev-<DATE>` is rebase-merged into `main`, and a new version tag is created.

### 🔥 Hotfixes
- Hotfixes are branched directly from `main` and squash-merged back into it.
- The same fix is **cherry-picked into `dev-<DATE>`** to ensure it’s preserved in the next release.
- Any active feature branches may need to rebase on the updated `dev-<DATE>`.

## ✅ How to Contribute

### 1. Pick a Feature
Check out the [Feature Wishlist](./docs/WISHLIST.md) in the repo. These are tasks I’ve curated and scoped out. Feel free to comment or reach out to say which one you’d like to work on.

If you have an idea that isn’t listed, feel free to open an issue or start a conversation.

### 2. Branch from the Current Dev Cycle
- Make sure you’re branching from the latest `dev-<DATE>` branch.
- Use a clear branch name, like `feature/summary-fig-obj-fft` or `bugfix/fix-init-print-unit`.

### 3. Keep Commits Focused
- Use small commits during development.
- Your PR will be squash-merged into a single feature commit.

### 4. Submit a PR
- Target your PR at the current `dev-<DATE>` branch.
- Use a descriptive PR title and include any relevant context.

I’ll review it, possibly ask for changes, and merge it once we’re both happy.

## 🧼 Coding Standards

- Follow existing code style conventions.
- Add docstrings to all public functions/classes.
- Include tests where relevant — even basic ones help.

## 📦 Dependencies & Setup

Setup instructions are available in the [README.md](./README.md). If you run into trouble setting up the environment, feel free to ask!

---

Thanks again for contributing — I’m looking forward to work with you on making PtyRAD even better! 🚀