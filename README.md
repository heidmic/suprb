# SupRB-2

## What is SupRB-2?

SupRB-2 is a Learning Classifier System (LCS) that utilises two separate optimizers that discover new rules and create global solutions to the learning problem, respectively. Those rules are stored in a Pool that expands during runtime and provides rules for solution creation.

<object data="docs/classifier.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="docs/classifier.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="docs/classifier.pdf">Download PDF</a>.</p>
    </embed>
</object>


## Install all requirements


    pip3 install -r suprb2/requirements.txt


Tested with Python 3.8.5 and Python 3.7.3.


## Contributing

A newly created branch must follow the following convention:

**<affiliation>_<Issue_number>\_<short_description_of_PR>**

e.g. oc_3_introduce_volume_for_rule_discovery


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
