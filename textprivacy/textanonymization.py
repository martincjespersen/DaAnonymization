"""Main module."""

from typing import List, Dict, Union, Set, Callable
import os
from sys import platform
import logging
import dacy

import re
import spacy
import torch
import torch.nn as nn
import multiprocessing
import numpy as np

spacy.prefer_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if platform == "linux" or platform == "linux2" or platform == "darwin":
    multiprocessing.set_start_method("fork")
elif platform == "win32":
    multiprocessing.set_start_method("spawn")


######### DaCy multiprocessing hack START #########
# Hack to make DaCy multiprocessable for both spawn and fork (SpaCy 3.0 issue with pickle)
torch.set_num_threads(1)
num_cpus: int = int(os.cpu_count())  # type: ignore
# path = os.path.dirname(os.path.realpath(__file__)).replace("textprivacy", "")
# dacy_path: str = os.environ.get(
#     "DACY",
#     "{}/da_dacy_large_tft-0.0.0/da_dacy_large_tft/da_dacy_large_tft-0.0.0".format(path),
# )
# print(dacy_path)
# print(os.path.dirname(os.path.realpath(__file__)))
# ner_model: nn.Module = None
# if os.path.exists(dacy_path):
#     print("here")
#     ner_model = spacy.load(dacy_path)  # type: ignore
ner_model = dacy.load("da_dacy_large_tft-0.0.0")


def worker(text: List[str]):  # type: ignore
    return list(ner_model.pipe(text, batch_size=len(text)))


######### DaCy multiprocessing hack END #########


class TextAnonymizer(object):
    """
    Object of a text corpus to apply masking function for anonymization

    Args:
        corpus: The corpus containing a list of strings
        mask_misc: Enable masking of miscellaneous entities (covers entities such as titles, events, religion etc.)
        suppression: Whether to suppress all entities with XXX
        individuals: Preset known individuals as a dict of dicts of dicts for specifying text index, person index and entities. For example:
                    individuals = { 100: {'PER': {'Martin Jespersen', 'Martin', 'Jespersen, Martin'} } }

    """

    def __init__(
        self,
        corpus: List[str] = [],
        mask_misc: bool = False,
        suppression: bool = False,
        individuals: Dict[int, Dict[str, Set[str]]] = {},
    ):
        super(TextAnonymizer, self).__init__()
        self.corpus = corpus
        self.mask_misc = mask_misc
        self.suppression = suppression
        self.individuals = individuals
        self.transformed_corpus: List[str]
        self.mapping: Dict[str, str] = {
            "PER": "[PERSON]",
            "LOC": "[LOKATION]",
            "ORG": "[ORGANISATION]",
            "CPR": "[CPR]",
            "TELEFON": "[TELEFON]",
            "EMAIL": "[EMAIL]",
        }
        self._supported_NE: List[str] = ["PER", "LOC", "ORG"]

        if self.mask_misc:
            self.mapping.update({"MISC": "DIVERSE"})
            self._supported_NE = ["PER", "LOC", "ORG", "MISC"]

        if self.suppression:
            self.mapping = {key: "XXX" for key in self.mapping}

    def find_cpr(self, text: str) -> Set[str]:
        """
        Find CPR numbers from a text

        Args:
            text: Text to remove CPR numbers from

        Returns:
            A set of CPR entities

        """
        cpr_pattern = "|".join(
            [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
        )
        cprs = set(re.findall(cpr_pattern, text))
        return cprs

    def find_telefon_nr(self, text: str) -> Set[str]:
        """
        Find telephone numbers from a text

        Args:
            text: Text to remove telephone numbers from

        Returns:
            A set of telephone number entities

        """
        tlf_pattern = "|".join(
            [
                r"\+\d{10}",
                r"\+\d{4} \d{2}​ \d{2}​ \d{2}",
                r"\+\d{2} \d{8}",
                r"\+\d{2} \d{2}​ \d{2}​ \d{2}​ \d{2}",
                r"\d{8}",
                r"\d{2}​ \d{2}​ \d{2}​​ \d{2}",
            ]
        )
        tlf_nrs = set(re.findall(tlf_pattern, text))
        return tlf_nrs

    def find_email(self, text: str) -> Set[str]:
        """
        Find emails from a text

        Args:
            text: Text to remove emails from

        Returns:
            A set of email entities

        """
        mail_pattern = r"[\w\.-]+@[\w\.-]+(?:\.[\w]+)+"
        emails = set(re.findall(mail_pattern, text))
        return emails

    def mask_entities(
        self, text: str, entities: Set[str], ent_type: str, suffix: str = ""
    ) -> str:
        """
        Masks a given entity from a text

        Args:
            text: Text to remove emails from
            entities: Set of entities to remove
            ent_type: Type of entity to determine placeholder to replace entity with
            suffix: A suffix to the placeholder (e.g., Person X)

        Returns:
            A text with the entity masked

        """
        sorted_entities = sorted(entities, key=len, reverse=True)
        for ent in sorted_entities:
            text = text.replace(ent, "{}{}".format(self.mapping[ent_type], suffix))
        return text

    """
    ################## Helper functions #################
    """

    def _apply_masks(
        self,
        text: str,
        methods: Dict[str, Callable],
        masking_order: List[str],
        ner_entities: Dict[str, Set[str]],
        index: int,
    ) -> str:
        """
        Masks a a set of entity types from a text

        Args:
            text: Text to mask entities from
            methods: A dictionary of masking methods to apply
            masking_order: The order of applying masking functions
            ner_entities: A dictiornary of lists containing the named entities found with DaCy
            index: Index of the text's placement in corpus

        Returns:
            A text with the a set of entity types masked

        """

        current_individuals = self.individuals.get(index, {})

        for method in masking_order:
            if method != "NER" and method in self.mapping:
                method_entitites = methods[method](text)
                method_entitites = method_entitites.union(
                    current_individuals.get(method, set([]))
                )
                text = self.mask_entities(text, method_entitites, method)
            else:
                # Handle DaCy entities
                for ent_name in ner_entities:
                    if ent_name in self.mapping:
                        rm_ents = ner_entities[ent_name].union(
                            current_individuals.get(ent_name, set())
                        )
                        text = self.mask_entities(text, rm_ents, ent_name)

                        if ent_name == "PER" and len(rm_ents) == 0:
                            logging.warning(
                                f"No person found in text at index {index} of text corpus"
                            )

        return text

    def _batch_prediction_DaCy(
        self, batch_size: int, n_process: int
    ) -> List[Dict[str, Set[str]]]:
        """
        Runs DaCy NER model on full corpus in batch mode and masks entities

        Args:
            batch_size: Number of texts to include in a batch
            n_process: Number of CPU cores to split computational on

        Returns:
            None

        """

        if device != "cuda":
            batches = (
                self.corpus[pos : pos + batch_size]
                for pos in range(0, len(self.corpus), batch_size)
            )
            with multiprocessing.Pool(n_process) as p:
                results = p.map(worker, batches)

            results = [item for sublist in results for item in sublist]
        else:
            results = ner_model.pipe(self.corpus, batch_size)

        entities: List[Dict[str, Set[str]]] = list()

        for i, doc in enumerate(results):
            text_entities: Dict[str, Set[str]] = {
                x: set([]) for x in self._supported_NE
            }
            for ent in doc.ents:
                if ent.label_ in text_entities:
                    text_entities[ent.label_].add(ent.text)

            entities.append(text_entities)

        return entities

    """
    ########## Mask multiple types of entities ##########
    """

    def mask_corpus(
        self,
        masking_order: List[str] = ["CPR", "TELEFON", "EMAIL", "NER"],
        custom_functions: Dict[str, Callable] = {},
        batch_size: int = 8,
        n_process: int = num_cpus,
        logging_file: str = None,
        loglevel: str = "DEBUG",
    ) -> List[str]:
        """
        Mask a corpus of danish text with provided methods

        Args:
            masking_order: Directed list of masking methods to apply to the corpus
            custom_functions: Dictionary containing custom masking functions as values and their names as keys
            batch_size: Used for DaCy running in batch mode
            n_process: Number of CPU cores to split computational on
            logging_file: Save log to file
            loglevel: Logging level to include in logging (default debug: include all)

        Returns:
            Anonymized version of the corpus

        """
        log_level = getattr(logging, loglevel.upper(), None)
        if logging_file:
            logging.basicConfig(
                filename=logging_file,
                filemode="w",
                format="%(asctime)s - %(levelname)s: %(message)s",
                datefmt="%d-%b-%y %H:%M:%S",
                level=log_level,
            )
        else:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s: %(message)s",
                datefmt="%d-%b-%y %H:%M:%S",
                level=log_level,
            )
        logging.info("##### Starting masking corpus #####")
        logging.info(f"Texts within corpus: {len(self.corpus)}")
        logging.info(f"Batch size for DaCy: {batch_size}")
        logging.info(f"Number of processes: {n_process}")

        methods = {
            "CPR": self.find_cpr,
            "TELEFON": self.find_telefon_nr,
            "EMAIL": self.find_email,
        }

        methods.update(custom_functions)

        if "NER" in masking_order:
            logging.info("Running DaCy Named Entity Recognition...")
            entities = self._batch_prediction_DaCy(batch_size, n_process)
            logging.info("Finished DaCy...")
        else:
            entities = [{} for x in self.corpus]

        self.transformed_corpus = []
        logging.info("Starting masking...")
        for i, text in enumerate(self.corpus):
            text = self._apply_masks(text, methods, masking_order, entities[i], i)
            self.transformed_corpus.append(text)

        logging.info("##### Completed masking! #####")
        return self.transformed_corpus
