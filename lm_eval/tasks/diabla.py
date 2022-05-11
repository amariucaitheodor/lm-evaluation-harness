# TODO: Remove all TODO comments once the implementation is complete.
"""
DiaBLa: English-French Bilingual dialogue dataset for Machine Translation
https://link.springer.com/article/10.1007/s10579-020-09514-4

Rachel Bawden, Eric Bilinski, Thomas Lavergne and Sophie Rosset
(2021). DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues
for Machine Translation. Language Resources and Evaluation(55). Pages
635–660. Springer Verlag. 10.1007/s10579-020-09514-4.

DiaBLa is an English-French dataset for the evaluation of Machine
Translation (MT) for informal, written bilingual dialogue.  It
contains 144 spontaneous dialogues (5,700+ sentences) between native
English and French speakers, mediated by one of two neural MT systems
in a range of role-play settings. The dialogues are accompanied by
fine-grained sentence-level judgments of MT quality, produced by the
dialogue participants themselves, as well as by manually normalised
versions and reference translations produced a posteriori

Homepage: http://almanach.inria.fr/software_and_resources/custom/DiaBLa-en.html
"""
from lm_eval.base import PromptSourceTask
from typing import List

_CITATION = """@article{bawden_DiaBLa:-A-Corpus-of_2021,
  author = {Bawden, Rachel and Bilinski, Eric and Lavergne, Thomas and Rosset, Sophie},
  doi = {10.1007/s10579-020-09514-4},
  title = {DiaBLa: A Corpus of Bilingual Spontaneous Written Dialogues for Machine Translation},
  year = {2021},
  journal = {Language Resources and Evaluation},
  publisher = {Springer Verlag},
  volume = {55},
  pages = {635--660},
  url = {https://hal.inria.fr/hal-03021633},
  pdf = {https://hal.inria.fr/hal-03021633/file/diabla-lre-personal-formatting.pdf},
}
"""


class DiaBLa(PromptSourceTask):
    VERSION = 0
    DATASET_PATH = "rbawden/DiaBLa"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def max_generation_length(self):
        return 512

    def stopping_criteria(self):
        return "\n"

    def invalid_doc_for_prompt(self, doc) -> bool:
        # Skip docs with empty references
        if self.doc_to_target(doc) in ([""], ""):
            return True
        return False

    def doc_to_target(self, doc) -> List[str]:
        _, target = self.prompt.apply(doc)
        if isinstance(target, list):
            return target
        else:
            return [target]
