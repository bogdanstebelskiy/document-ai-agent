# Git

A Git commit records a snapshot of the project in the repository history. `git add` puts changes into the staging area; it does not create a commit.

A branch is a movable reference to a commit. Creating a new branch does not copy the whole repository. It gives you another name that can point to commits as you work.

`git restore` can discard changes in the working tree, while `git restore --staged` removes a file from the staging area without necessarily discarding its contents.

A merge combines histories. A rebase instead moves a series of commits onto a different base, producing a new history. Rebasing local work can make a branch history easier to follow, but rewriting shared history can cause problems.
