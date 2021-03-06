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
    """docstring for TextAnonymizer"""

    def __init__(self, corpus: List[str], context_specific: bool = False):
        super(TextAnonymizer, self).__init__()
        self.corpus = corpus
        self.context_specific = context_specific
        self.entities: Dict[Union[str, int], Union[str, int]] = dict()
        self.ner_model: nn.Module
        self.nlp: Callable
        self.ner_type: str = ""

    @staticmethod
    def mask_cpr(text: str) -> str:
        cpr_pattern = "|".join(
            [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
        )
        cprs = re.findall(cpr_pattern, text)
        for cpr in cprs:
            text = text.replace(cpr, "[CPR]")

        return text

    @staticmethod
    def mask_telefon_nr(text: str) -> str:
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
        mail_pattern = r"[\w\.-]+@[\w\.-]+(?:\.[\w]+)+"
        emails = re.findall(mail_pattern, text)
        for email in emails:
            text = text.replace(email, "[EMAIL]")

        return text

    def mask_NER(self, text: str, max_len: int = 250) -> str:
        mapping: Dict[Union[str, int], str] = {
            "PER": "PERSON",
            "LOC": "LOKATION",
            "ORG": "ORGANISATION",
        }

        self._run_NER(text, max_len)

        for entity_text, entity in self.entities.items():
            if entity in mapping:
                text = text.replace(str(entity_text), "[{}]".format(mapping[entity]))
        return text

    """
    ################## Helper functions #################
    """

    def _load_NER_model(self, NER_type: str = "danlp") -> None:
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

    def _update_entities(self, entity_labels: BERT_output) -> None:

        for entity in entity_labels["entities"]:
            self.entities[entity["text"]] = entity["type"]

    def _run_NER(self, text: str, max_len: int) -> None:
        # Avoid using too much memory (and bert has maximum tokens of 512)
        sentence = self.nlp(text)
        sentence_chunks = [
            list(sentence)[x : x + max_len]
            for x in range(0, len(list(sentence)), max_len)
        ]
        for chunk in sentence_chunks:
            if self.ner_type == "danlp":
                e_lab = self.ner_model.predict([x.text for x in chunk], IOBformat=False)
                self._update_entities(e_lab)
            elif self.ner_type == "dacy":
                doc = self.ner_model(" ".join([x.text for x in chunk]))
                chunk_e_lab: Dict[Union[str, int], Union[str, int]] = {
                    ent.text: ent.label_ for ent in doc.ents
                }
                self.entities.update(chunk_e_lab)
            else:
                raise Exception("Not implemented: {}".format(self.ner_type))

    """
    ########## Mask multiple types of entities ##########
    """

    def mask_corpus(
        self, masking_methods: List[str] = ["cpr", "telefon", "email", "NER"]
    ) -> List[str]:
        methods = {
            "cpr": self.mask_cpr,
            "telefon": self.mask_telefon_nr,
            "email": self.mask_email,
            "NER": self.mask_NER,
        }

        for method in masking_methods:
            self.corpus = list(map(methods[method], self.corpus))  # type: ignore

        return self.corpus
