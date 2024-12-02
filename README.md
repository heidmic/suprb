[![DOI](https://zenodo.org/badge/303331999.svg)](https://zenodo.org/badge/latestdoi/303331999)

# SupRB

## What is SupRB?

SupRB is a rule-set learning algorithm (or Learning Classifier System (LCS)) that utilises two separate optimizers that discover new rules and create global solutions to the learning problem, respectively. Those rules are stored in a Pool that expands during runtime and provides rules for solution creation.

<!---
![SupRB statemachine](./docs/suprb.png)
-->

<img src="./docs/suprb.png" alt="SupRB statemachine" width="500"/>

Rules use interval bounds for matching and a simple local model for outputs.

<img src="./docs/rule.png" alt="Rule" width="700"/>

Solutions of the problem select classifiers from the unchanging pool.

<img src="./docs/solution.png" alt="Solutions and Pool of classifiers" width="500"/>

## Usage

For a quick entry into to using SupRB, have a look at the **examples** directory or the [suprb-experimentation repository](https://github.com/heidmic/suprb-experimentation) for more complex setups including automated parametertuning.

The most basic way of using SupRB is by specifying the *rule_generation* and *solution_composition* with its default parameters: 
`SupRB(rule_generation=ES1xLambda(), solution_composition=GeneticAlgorithm())`

Most of the time you want to use **cross_validate** from **sklearn.model_selection** for your model to get a better idea of general performance, which is why the examples use it consistently.

Make sure that the data you feed the model is in the correct format (use the examples for reference):
- Inputs/features should be min-max-scaled to [-1, 1].
- Output/target should be standardized (mean to zero, variance to one).

The examples in the examples directory are:
- [example_1.py](https://github.com/heidmic/suprb/blob/master/examples/example_1.py): Using default parameters
- [example_2.py](https://github.com/heidmic/suprb/blob/master/examples/example_2.py): With one level of changed parameters
- [example_3.py](https://github.com/heidmic/suprb/blob/master/examples/example_3.py): With two levels of changed parameters
- [example_4.py](https://github.com/heidmic/suprb/blob/master/examples/example_4.py): Without any default parameters
- [compare_sklearn.py](https://github.com/heidmic/suprb/blob/153_add_usage_guide/examples/compare_sklearn.py): Code to run a direct comparison between suprb and various other sklearn models


## Install all requirements


    pip3 install -r requirements.txt


Tested with Python 3.9.4.


## Contributing

Newly created branch should follow a pattern that allows easy comprehension of what is being done and why, e.g.:

**<Issue_number>\_<short_description_of_PR>**, e.g. *3_introduce_volume_for_rule_discovery*.

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

## Publications

### The Concept

Michael Heider, Helena Stegherr, Jonathan Wurth, Roman Sraj, and Jörg Hähner. 2022. **Separating Rule Discovery and Global Solution Composition in a Learning Classifier System.** In Genetic and Evolutionary Computation Conference Companion (GECCO ’22 Companion). https://doi.org/10.1145/3520304.3529014

### Comparisons with other Systems

Michael Heider, Helena Stegherr, Jonathan Wurth, Roman Sraj, and Jörg Hähner. 2022. **Investigating the Impact of Independent Rule Fitnesses in a Learning Classifier System.** In: Mernik, M., Eftimov, T., Črepinšek, M. (eds) Bioinspired Optimization Methods and Their Applications. BIOMA 2022. Lecture Notes in Computer Science, vol 13627. Springer, Cham. https://doi.org/10.1007/978-3-031-21094-5_11 http://arxiv.org/abs/2207.05582 A comparison with XCSF.

Michael Heider, Helena Stegherr, Roman Sraj, David Pätzel, Jonathan Wurth and Jörg Hähner. 2023. **SupRB in the context of rule-based machine learning methods: a comparative study.** Applied Soft Computing 147, 110706. DOI: 10.1016/j.asoc.2023.110706 A comparison with XCSF, Decision Trees, and Random Forests.

### Investigating Components

Jonathan Wurth, Michael Heider, Helena Stegherr, Roman Sraj, and Jörg Hähner. 2022. **Comparing different Metaheuristics for Model Selection in a Supervised Learning Classifier System.** In Genetic and Evolutionary Computation Conference Companion (GECCO ’22 Companion). https://doi.org/10.1145/3520304.3529015

Michael Heider, Helena Stegherr, David Pätzel, Roman Sraj, Jonathan Wurth and Jörg Hähner. 2022. **Approaches for Rule Discovery in a Learning Classifier System.** In Proceedings of the 14th International Joint Conference on Computational Intelligence - ECTA. https://doi.org/10.5220/0011542000003332

Michael Heider, Helena Stegherr, David Pätzel, Roman Sraj, Jonathan Wurth, Benedikt Volger and Jörg Hähner. 2023. **Discovering rules for rule-based machine learning with the help of novelty search.** SN Computer Science 4, 6, 778. DOI: 10.1007/s42979-023-02198-x

Michael Heider, Maximilian Krischan, Roman Sraj and Jörg Hähner. 2024 **Exploring Self-Adaptive Genetic Algorithms to Combine Compact Sets of Rules.** In Proceedings of the 2024 IEEE Congress on Evolutionary Computation (CEC). https://doi.org/10.1109/CEC60901.2024.10612101.

### Explainability of LCS

Michael Heider, Helena Stegherr, Richard Nordsieck and Jörg Hähner. 2023. **Assessing model requirements for explainable AI: a template and exemplary case study.** Artificial Life 29, 4, 468-486. MIT press. DOI: https://doi.org/10.1162/artl_a_00414

### A general description of LCS and its optimizers

Michael Heider, David Pätzel, Helena Stegherr, Jörg Hähner. 2023. **A Metaheuristic Perspective on Learning Classifier Systems.** In: Eddaly, M., Jarboui, B., Siarry, P. (eds) Metaheuristics for Machine Learning. Computational Intelligence Methods and Applications. Springer, Singapore. https://doi.org/10.1007/978-981-19-3888-7_3


## Experiments

We provide the experiment scripts used in associated publications under: https://github.com/heidmic/suprb-experimentation


## Citation

If you use this code for research, please cite:

Michael Heider, Helena Stegherr, Jonathan Wurth, Roman Sraj, and Jörg Hähner. 2022. **Separating Rule Discovery and Global Solution Composition in a Learning Classifier System.** In Genetic and Evolutionary Computation Conference Companion (GECCO ’22 Companion). https://doi.org/10.1145/3520304.3529014
