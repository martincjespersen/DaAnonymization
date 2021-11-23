"""Main module."""

from typing import List, Dict, Set, Callable
from textprivacy.textanonymization import TextAnonymizer
from textprivacy.utils import is_valid_number, get_integer, get_float, laplace_noise

import logging


class TextPseudonymizer(TextAnonymizer):
    """
    Object of a text corpus to apply masking function for pseudonymization

    Args:
        corpus: The corpus containing a list of strings
        mask_misc: Enable masking of miscellaneous entities (covers entities such as titles, events, religion etc.)
        individuals: Preset known individuals as a dict of dicts of dicts for specifying text index, person index and entities. For example:
                    individuals = { 100: {1: {'PER': {'Martin Jespersen', 'Martin', 'Jespersen, Martin'} } }}

    """

    def __init__(
        self,
        corpus: List[str] = [],
        mask_misc: bool = False,
        individuals: Dict[int, Dict[int, Dict[str, Set[str]]]] = {},
        mask_numbers: bool = False,
        epsilon: float = None,
    ):
        super(TextPseudonymizer, self).__init__(corpus, mask_misc, False)
        self.individuals = individuals  # type: ignore
        self.mask_numbers = mask_numbers
        self.epsilon = epsilon
        self.mapping: Dict[str, str] = {
            "PER": "Person",
            "LOC": "Lokation",
            "ORG": "Organisation",
            "CPR": "CPR",
            "TELEFON": "Telefon",
            "EMAIL": "Email",
        }

        self._supported_NE: List[str] = ["PER", "LOC", "ORG"]

        if self.mask_misc:
            self.mapping.update({"MISC": "Diverse"})
            self._supported_NE.append("MISC")

        if self.mask_numbers:
            self.mapping.update({"NUM": "Nummer"})
            self._supported_NE.append("NUM")

    """
    ################## Helper functions #################
    """

    def _update_entity(
        self,
        entities: Set[str],
        current_individuals: Dict[int, Dict[str, Set[str]]],
        entity_type: str,
    ) -> Dict[int, Dict[str, Set[str]]]:

        """
        Updates and pairs of the entity to individuals

        Args:
            entities: A set of entities identified
            current_individuals: Dictionary of current individuals and their identified entities
            entity_type: Current entity to update and pair to individuals

        Returns:
            A dictionary of current individuals and their entities

        """

        sorted_entities = sorted(entities, key=len, reverse=True)
        n_individals: int = 0
        if current_individuals:
            n_individals = max(current_individuals.keys())

        while sorted_entities:
            entity = sorted_entities.pop(0)

            flag = False
            for individual in current_individuals:
                if entity_type in current_individuals[individual] and any(
                    entity.lower() in e.lower()
                    for e in current_individuals[individual][entity_type]
                ):
                    flag = True
                    current_individuals[individual][entity_type].add(entity)

            if not flag:
                n_individals += 1
                current_individuals[n_individals] = {x: set() for x in self.mapping}
                current_individuals[n_individals][entity_type].add(entity)

        return current_individuals

    def _update_individuals(
        self, all_entities: Dict[str, Set[str]], index: int
    ) -> Dict[int, Dict[str, Set[str]]]:
        """
        Updates all types of entities for all individuals in a text

        Args:
            all_entities: A dictionary of all entities found in the text
            index: The index of the current text's placement in corpus

        Returns:
            A dictionary of current individuals and their entities

        """
        current_individuals = self.individuals.get(index, {})
        order_entities = sorted(list(all_entities.keys()))
        order_entities.pop(order_entities.index("PER"))
        order_entities.insert(0, "PER")

        for ent in order_entities:
            if all_entities[ent]:
                current_individuals = self._update_entity(  # type: ignore
                    all_entities[ent], current_individuals, ent  # type: ignore
                )

        return current_individuals  # type: ignore

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
        all_entities: Dict[str, Set[str]] = {}
        for method in masking_order:
            if method != "NER":
                entities = methods[method](text)
                all_entities[method] = entities
            else:
                # Handle DaCy entities
                all_entities.update(ner_entities)

        individuals = self._update_individuals(all_entities, index)
        self.individuals[index] = individuals  # type: ignore
        total_people = 0

        # get all entities into one list
        masked_entities: List[List[str]] = []
        for person in sorted(individuals):
            suffix = " {}".format(person)
            for method in masking_order:
                if method != "NER" and method in individuals[person]:
                    for ent in individuals[person][method]:
                        if [method, ent, suffix] not in masked_entities:
                            masked_entities.append([method, ent, suffix])
                else:
                    for ent_ in self._supported_NE:
                        if ent_ in individuals[person]:
                            for ent in individuals[person][ent_]:
                                if [ent_, ent, suffix] not in masked_entities:
                                    masked_entities.append([ent_, ent, suffix])

        # run through all entities, sorted in descending order of entity size
        masked_entities = sorted(masked_entities, key=lambda x: len(x[1]), reverse=True)
        for method, ent, suffix in masked_entities:
            if method == "NUM" and self.epsilon:
                text = self.noisy_numbers(
                    text,
                    set([ent]),
                    self.epsilon,
                    placeholder=self.mapping[method],
                    suffix=suffix,
                )
            else:
                text = self.mask_entities(text, set([ent]), method, suffix=suffix)

                if method == "PER":
                    total_people += 1

        if total_people == 0:
            logging.warning(f"No person found in text at index {index} of text corpus")

        return text
