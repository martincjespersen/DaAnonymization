"""Main module."""
import re


class TextAnonymizer(object):
    """docstring for TextAnonymizer"""

    def __init__(self, arg):
        super(TextAnonymizer, self).__init__()
        self.arg = arg

    @classmethod
    def mask_cpr(self, text: str) -> str:
        cpr_pattern = "|".join(
            [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
        )
        cprs = re.findall(cpr_pattern, text)
        for cpr in cprs:
            text.replace(cpr, "<CPR>")

        return text
