"""Main module."""

from typing import List, Dict, Union, TypedDict, Callable

from danlp.models import load_bert_ner_model
import re
import da_core_news_sm
import spacy
import torch.nn as nn


class BERT_output(TypedDict):
    entities: List[Dict[str, Union[int, str]]]
    text: str


class TextAnonymizer(object):
    """
    Object of a text corpus to apply masking function for anonymization

    Args:
        corpus: The corpus containing a list of strings
        context_specific: Boolean to determine if entities should be context specific or not
    """

    def __init__(self, corpus: List[str], context_specific: bool = False):
        super(TextAnonymizer, self).__init__()
        self.corpus = corpus
        self.context_specific = context_specific
        self.ner_model: nn.Module
        self.nlp: Callable
        self.ner_type: str = ""
        self.mapping: Dict[Union[str, int], str] = {
            "PER": "PERSON",
            "LOC": "LOKATION",
            "ORG": "ORGANISATION",
        }

        if self.context_specific:
            raise Exception("Context specific anonymization is not implemented yet ")

    @staticmethod
    def mask_cpr(text: str) -> str:
        """
        Masks CPR numbers from a text

        Args:
            text: Text to remove CPR numbers from

        Returns:
            Text with [CPR] instead of the CPR numbers

        """
        cpr_pattern = "|".join(
            [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
        )
        cprs = re.findall(cpr_pattern, text)
        for cpr in cprs:
            text = text.replace(cpr, "[CPR]")

        return text

    @staticmethod
    def mask_telefon_nr(text: str) -> str:
        """
        Masks telephone numbers from a text

        Args:
            text: Text to remove telephone numbers from

        Returns:
            Text with [TELEFON] instead of the telephone numbers

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
        tlf_nrs = re.findall(tlf_pattern, text)
        for tlf_nr in tlf_nrs:
            text = text.replace(tlf_nr, "[TELEFON]")

        return text

    @staticmethod
    def mask_email(text: str) -> str:
        """
        Masks emails from a text

        Args:
            text: Text to remove emails from

        Returns:
            Text with [EMAIL] instead of the email adress

        """
        mail_pattern = r"[\w\.-]+@[\w\.-]+(?:\.[\w]+)+"
        emails = re.findall(mail_pattern, text)
        for email in emails:
            text = text.replace(email, "[EMAIL]")

        return text

    def mask_NER(self, text: str, max_len: int = 250) -> str:
        """
        Masks named entities (person, location and organization) from a text

        Args:
            text: Text to remove entities from

        Returns:
            Text with entity specific token (e.g., person = [PER]) instead of the entity

        """
        entities = self._run_NER(text, max_len)

        for entity_text, entity in entities.items():
            if entity in self.mapping:
                text = text.replace(
                    str(entity_text), "[{}]".format(self.mapping[entity])
                )
        return text

    """
    ################## Helper functions #################
    """

    def _load_NER_model(self, NER_type: str = "danlp") -> None:
        """
        Load NER model

        Args:
            NER_type: Which type to load (danlp or dacy)

        Returns:
            None

        """
        self.nlp = da_core_news_sm.load()
        if NER_type == "danlp":
            self.ner_model = load_bert_ner_model()
            self.ner_type = "danlp"
            print(type(self.nlp))
        elif NER_type == "dacy":
            self.ner_model = spacy.load(
                "da_dacy_large_tft-0.0.0/da_dacy_large_tft/da_dacy_large_tft-0.0.0"
            )
            self.ner_type = "dacy"
        else:
            raise Exception("Not implemented: {}".format(NER_type))

    def _update_entities(
        self,
        entities: Dict[Union[str, int], Union[str, int]],
        entity_labels: BERT_output,
    ) -> None:
        """
        Update current entities with new predicted entities

        Args:
            entity_labels: DaNLP's bert output for predicted entities

        Returns:
            None

        """
        for entity in entity_labels["entities"]:
            entities[entity["text"]] = entity["type"]

    def _run_NER(
        self, text: str, max_len: int
    ) -> Dict[Union[str, int], Union[str, int]]:
        """
        Runs NER model on a text entry

        Args:
            text: Text to predict named entities on
            max_len: Maximum tokens to include in prediction to fit BERT and memory issues

        Returns:
            None

        """
        entities: Dict[Union[str, int], Union[str, int]] = dict()
        sentence = self.nlp(text)
        sentence_chunks = [
            list(sentence)[x : x + max_len]
            for x in range(0, len(list(sentence)), max_len)
        ]
        for chunk in sentence_chunks:
            if self.ner_type == "danlp":
                e_lab = self.ner_model.predict([x.text for x in chunk], IOBformat=False)
                self._update_entities(entities, e_lab)
            elif self.ner_type == "dacy":
                doc = self.ner_model(" ".join([x.text for x in chunk]))
                chunk_e_lab: Dict[Union[str, int], Union[str, int]] = {
                    ent.text: ent.label_ for ent in doc.ents
                }
                entities.update(chunk_e_lab)
            else:
                raise Exception("Not implemented: {}".format(self.ner_type))

        return entities

    def _batch_prediction_DaCy(self, batch_size: int):

        assert self.ner_type == "dacy", "DaCy NER model not set"

        docs = self.ner_model.pipe(self.corpus, batch_size=batch_size)
        for i, doc in enumerate(docs):
            for ent in doc.ents:
                if ent.label_ in self.mapping:
                    self.corpus[i] = self.corpus[i].replace(
                        ent.text, "[{}]".format(self.mapping[ent.label_])
                    )

    """
    ########## Mask multiple types of entities ##########
    """

    def mask_corpus(
        self,
        masking_methods: List[str] = ["cpr", "telefon", "email", "NER"],
        custom_functions: Dict[str, Callable] = {},
        batch_size: int = 8,
    ) -> List[str]:
        """
        Mask a corpus of danish text with provided methods

        Args:
            masking_methods: Directed list of masking methods to apply to the corpus
            custom_functions: Dictionary containing custom masking functions as values and their names as keys

        Returns:
            Anonymized version of the corpus

        """

        methods = {
            "cpr": self.mask_cpr,
            "telefon": self.mask_telefon_nr,
            "email": self.mask_email,
            "NER": self.mask_NER,
        }

        methods.update(custom_functions)

        for method in masking_methods:
            if method == "NER" and self.ner_type == "dacy":
                continue
            self.corpus = list(map(methods[method], self.corpus))  # type: ignore

        if self.ner_type == "dacy":
            self._batch_prediction_DaCy(batch_size)

        return self.corpus
