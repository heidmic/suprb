# SupRB2

## What is SupRB2?

SupRB2 is a Learning Classifier System (LCS) that utilises two separate optimizers that discover new rules and create global solutions to the learning problem, respectively. Those rules are stored in a Pool that expands during runtime and provides rules for solution creation.

![SupRB2 statemachine](./docs/suprb2.png)

Classifiers or Rules use interval bounds for matching and a simple local model for outputs.

![Classifier/Rule](./docs/rule.png)

Solutions of the problem select classifiers from the unchanging pool.

<img src="./docs/solution.png" alt="Solutions and Pool of classifiers" width="600"/>


## Install all requirements


    pip3 install -r requirements.txt


Tested with Python 3.9.4.


## Contributing

Newly created branch should follow a pattern that allows easy comprehension of what is being done and why, e.g.:

**<Issue_number>\_<short_description_of_PR>**, e.g. *issue_3_introduce_volume_for_rule_discovery*.

If there is no open issue the branch name should just reflect a short description of the feature/fix, e.g. *introduce_volume_for_rule_discovery*.


The commit messages of all the commits in your PR should be properly formatted:
- The first line is the commit message's *summary line*.
- The summary line starts with a *capitalized imperative* (e.g. “Add …”, “Fix
  …”, “Merge …”).
- The summary line is just one line (aim for 50 characters or less).
- The summary line is a heading—it *doesn't* end with a period!
- If more text follows (which will often be the case) there *must* be one blank
  line after the summary line.
- All lines after the summary line (the commit message's *body*) should be
  wrapped at around 72 characters.  Remember to leave a blank line after the
  summary line.
- Use the message body for more detailed explanations if necessary. Don't put
  these in the summary line.
- The body may contain multiple paragraphs, separated by *blank lines*.
- The body may contain bullet lists.
- Basically, adhere to
  [this](https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

Avoid merge commits by using rebase rather than merge when combining branches
