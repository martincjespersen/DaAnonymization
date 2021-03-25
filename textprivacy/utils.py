from typing import Set, Union, Tuple

import re
import numpy as np


def is_valid_number(number: str) -> str:
    """
    Determines whether the number is a valid number

    Args:
        number: A string of the identified number

    Returns:
        A string whether it is considered a float, integer or invalid

    """
    dots = number.count(".")
    commas = number.count(",")
    reversed_num = number[::-1]
    end_dot = reversed_num.index(".") if dots > 0 else 1000
    end_comma = reversed_num.index(",") if commas > 0 else 1000
    last_delimiter = end_dot - end_comma

    special_characters = ":;!@#$%^&*()+?_=<>/"

    if (dots == 1 and last_delimiter < 0) or (commas == 1 and last_delimiter > 0):
        return "float"
    elif (
        (dots > 1 and commas > 1)
        or (dots == 1 and last_delimiter > 0)
        or (commas == 1 and last_delimiter < 0)
        and not any(x in number for x in special_characters)
    ):
        return "invalid"

    return "integer"


def get_integer(number: str) -> int:
    """
    Fetches all numbers and returns an integer

    Args:
        number: A string of the identified number

    Returns:
        An integer value of number

    """
    return int("".join(re.findall(r"\d+", number)))


def get_float(number: str) -> Tuple[float, int, str]:
    """
    Returns a proper float value

    Args:
        number: A string of the identified number

    Returns:
        A tuple of the float, the float precision and sign

    """
    dots = number.count(".")
    commas = number.count(",")
    reversed_num = number[::-1]

    if dots > 0 and commas > 0:
        end_dot = reversed_num.index(".")
        end_comma = reversed_num.index(",")

        float_delimiter = "." if end_dot - end_comma < 0 else ","
    else:
        float_delimiter = (
            "." if number.count(".") == 1 else "," if number.count(",") == 1 else ""
        )

    if float_delimiter == ",":
        number = number.replace(".", "").replace(",", ".")

    sign = "-" if number.startswith("-") else ""
    number_s = sign + "".join(re.findall(r"[\d.]+", number))
    precision = len(number_s) - number_s.index(".") - 1 if float_delimiter != "" else 0
    f_number: float = float(number_s)

    return f_number, precision, sign


def laplace_noise(
    number: Union[float, int], epsilon: float, sign: str, integer: str
) -> Union[float, int]:
    """
    Adds laplace noise of (epsilon, 0) distribution

    Args:
        number: A string of the identified number
        epsilon: Parameter used for laplace distribution (similar to differential privacy)
        sign: The sign of the number
        integer: Whether number is an integer

    Returns:
        Returns the float or integer with added laplace noise

    """
    noise = np.random.laplace(0, 1.0 / epsilon, 1)[0]
    noisy_number = number + noise
    noisy_number = abs(noisy_number) if not sign else noisy_number
    if integer == "integer":
        noisy_number = int(noisy_number)
    return noisy_number
