# From course experiment to runnable repository

This is Waleed Siddique's DASC 5420 Theoretical Machine Learning project at Thompson Rivers University: “How Does Training Sample Size Affect the Generalization Performance of Machine Learning Classifiers?”

The public experiment is adapted from the existing DASC5420_Experiment_Code.py. The original numerical tables are retained as course_results.csv and course_plateaus.csv, alongside newly executed outputs in results/.

Publication changes add a main guard, output and worker arguments, a noninteractive plotting backend, explicit failure on unsuccessful fits, a reusable version of the existing diminishing returns calculation, environment recording, tests and CI. Defaults retain the dataset, models, random seeds, training fractions, five folds, scoring and sample standard deviations from the original experiment. One worker is the default so the run does not exhaust a laptop.

The course report and submission forms contain personal administrative details and are not included. The original report's disclosure is preserved here: ChatGPT was used for brainstorming, language editing, code troubleshooting, interpretation support, and document layout assistance. Repository preparation also used code assistance for packaging and verification. This statement preserves the course disclosure requirement.

No claim is made that the repository has received a grade or instructor approval. The original source files remain unchanged outside this repository.
