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

from textprivacy.utils import is_valid_number, get_integer, get_float, laplace_noise

spacy.prefer_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        multiprocessing.set_start_method("fork")
except RuntimeError:
    pass
# elif platform == "win32":
#     multiprocessing.set_start_method("spawn")


######### DaCy multiprocessing hack START #########
# Hack to make DaCy multiprocessable for both spawn and fork (SpaCy 3.0 issue with pickle)
torch.set_num_threads(1)
num_cpus: int = int(os.cpu_count())  # type: ignore
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
        mask_numbers: bool = False,
        epsilon: float = None,
    ):
        super(TextAnonymizer, self).__init__()
        self.corpus = corpus
        self.mask_misc = mask_misc
        self.mask_numbers = mask_numbers
        self.epsilon = epsilon
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
            self.mapping.update({"MISC": "[DIVERSE]"})
            self._supported_NE.append("MISC")

        if self.mask_numbers:
            self.mapping.update({"NUM": "[NUMMER]"})
            self._supported_NE.append("NUM")

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
                r"\+\d{4} \d{2} \d{2} \d{2}",
                r"\+\d{2} \d{8}",
                r"\+\d{2} \d{2} \d{2} \d{2} \d{2}",
                r"\+\d{2} \d{4} \d{4}",
                r"\d{2} \d{4} \d{4}",
                r"\d{2} \d{4}\-\d{4}",
                r"\d{8}",
                r"\d{4} \d{4}",
                r"\d{4}\-\d{4}",
                r"\d{2} \d{2} \d{2} \d{2}",
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
            text: Text to remove entity from
            entities: Set of entities to remove
            ent_type: Type of entity to determine placeholder to replace entity with
            suffix: A suffix to the placeholder (e.g., Person X)

        Returns:
            A text with the entity masked

        """
        sorted_entities = sorted(list(set(entities)), key=len, reverse=True)
        for ent in sorted_entities:
            ent = ent.strip()
            if ent_type == "PER":
                ent = ent.replace(".", "")
            if ent != "" and not (
                ent_type in ["PER", "LOC", "ORG", "EMAIL", "CPR", "TELEFON"]
                and len(ent) <= 2
            ):
                ent_regex = r"(.?)({})(.?)".format(re.escape(ent))
                regexs = re.findall(ent_regex, text)

                # ensure entities are not subwords or subnumbers
                for reg_prefix, word, reg_suffix in regexs:

                    if not re.search(
                        r"[a-zæøåA-ZÆØÅ0-9]", reg_prefix
                    ) and not re.search(r"[a-zæøåA-ZÆØÅ0-9]", reg_suffix):
                        to_be_masked = re.escape(
                            r"{}{}{}".format(reg_prefix, word, reg_suffix)
                        )
                        to_mask = r"{}{}{}{}".format(
                            reg_prefix, self.mapping[ent_type], suffix, reg_suffix
                        )

                        text = re.sub(to_be_masked, to_mask, text)

        return text

    def noisy_numbers(
        self,
        text: str,
        entities: Set[str],
        epsilon: float,
        placeholder: str = "[NUMMER]",
        suffix: str = "",
    ) -> str:
        """
        Adds noises to numbers

        Args:
            text: Text to mask numbers from
            entities: Set of numbers to add noise or remove
            epsilon: Parameter used for laplace distribution (similar to differential privacy)
            placeholder: Fallback placeholder for invalid numbers
            suffix: Fallback suffix for pseudonymized numbers

        Returns:
            A text with the entity masked

        """

        tokens = ner_model.tokenizer(text)

        words = list()
        prev_word = ""
        for token in tokens:
            # avoid applying noise to pseudo identifiers
            if token.text not in entities or prev_word in self.mapping.values():
                prev_word = token.text
                words.append("{}{}".format(token.text, token.whitespace_))
                continue

            validity = is_valid_number(token.text)
            precision = None
            sign = ""
            if validity == "invalid":
                word = "{}{}{}".format(placeholder, suffix, token.whitespace_)
                prev_word = word
                words.append(word)
            elif validity == "float":
                value, precision, sign = get_float(token.text)
            else:
                value = get_integer(token.text)
            noisy_number = laplace_noise(value, epsilon, sign, validity)
            string_number = str(round(noisy_number, precision))
            word = "{}{}".format(string_number, token.whitespace_)
            prev_word = word
            words.append(word)

            prev_word = word

        return "".join(words)

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

                        if ent_name == "NUM" and self.epsilon:
                            text = self.noisy_numbers(
                                text,
                                rm_ents,
                                self.epsilon,
                                placeholder=self.mapping[ent_name],
                            )
                        else:
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

        if device != "cuda" and platform != "win32":
            # processes = n_process if n_process < len(self.corpus) else len(self.corpus)
            batches = (
                self.corpus[pos : pos + batch_size]
                for pos in range(0, len(self.corpus), batch_size)
            )
            with multiprocessing.Pool(n_process) as p:
                results = p.map(worker, batches)

            results = [item for sublist in results for item in sublist]
        else:
            torch.set_num_threads(n_process)
            results = ner_model.pipe(self.corpus, batch_size=batch_size)

        entities: List[Dict[str, Set[str]]] = list()

        for i, doc in enumerate(results):
            text_entities: Dict[str, Set[str]] = {
                x: set([]) for x in self._supported_NE
            }
            for ent in doc.ents:
                if ent.label_ in text_entities:
                    text_entities[ent.label_].add(ent.text)

            if "NUM" in self.mapping:
                # get numbers from part of speech tags
                for token in doc:
                    # ensure the number isn't in another NER token
                    digits = len([x for x in token.text if x.isdigit()])
                    if token.tag_ == "NUM" and not token.ent_type_ and digits > 0:
                        text_entities["NUM"].add(token.text)

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
        logging.info("##### General settings #####")
        logging.info(f"Texts within corpus: {len(self.corpus)}")
        logging.info(f"Batch size for DaCy: {batch_size}")
        logging.info(f"Number of processes: {n_process}")
        logging.info(f"Numerical Laplace epsilon: {self.epsilon}")
        logging.info(f"Suppression: {self.suppression}")

        methods = {
            "CPR": self.find_cpr,
            "TELEFON": self.find_telefon_nr,
            "EMAIL": self.find_email,
        }

        methods.update(custom_functions)

        entities_masked = [x for x in masking_order if x != "NER"]
        if "NER" in masking_order:
            entities_masked = entities_masked + self._supported_NE

        logging.info("Entities: {}".format(",".join(entities_masked)))

        logging.info("##### Starting masking corpus #####")
        if "NER" in masking_order:
            logging.info("Running DaCy Named Entity Recognition...")
            entities = self._batch_prediction_DaCy(batch_size, n_process)
            logging.info("Finished DaCy...")
        else:
            entities = [{} for x in self.corpus]

        self.transformed_corpus = []
        logging.info("Starting masking...")
        for i, text in enumerate(self.corpus):
            try:
                text = self._apply_masks(text, methods, masking_order, entities[i], i)
            except Exception as e:
                logging.critical(
                    f"Text at index {i} in corpus failed to be transformed with error: {str(e)}"
                )
                text = f"Text at index {i} in corpus failed to be transformed with error: {str(e)}"

            self.transformed_corpus.append(text)

        logging.info("##### Completed masking! #####")
        return self.transformed_corpus
