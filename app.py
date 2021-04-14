from flask import Flask, render_template, request, url_for, jsonify
from textprivacy import TextAnonymizer
from textprivacy import TextPseudonymizer

app = Flask(__name__)


def clean_individual_format(individuals):
    cleaned = {}
    for index in individuals:
        cleaned[int(index)] = {}
        for person in individuals[index]:
            cleaned[int(index)][int(person)] = {
                x: set(individuals[index][person][x])
                for x in individuals[index][person]
            }

    return cleaned


def clean_individual_format_anon(individuals):
    cleaned = {}
    for index in individuals:
        cleaned[int(index)] = {
            x: set(individuals[index][x]) for x in individuals[index]
        }

    return cleaned


@app.route("/pseudonymize", methods=["POST"])
def pseudonymize():
    if not request.json or not "rawtext_list" in request.json:
        abort(400)

    rawtext_list = request.json["rawtext_list"]
    individuals = clean_individual_format(request.json.get("individuals", {}))

    Pseudonymizer = TextPseudonymizer(rawtext_list, individuals=individuals)
    masked_corpus = Pseudonymizer.mask_corpus()

    response = {
        "output": masked_corpus,
    }

    return jsonify(response), 200


@app.route("/anonymize", methods=["POST"])
def anonymize():
    if not request.json or not "rawtext_list" in request.json:
        abort(400)

    rawtext_list = request.json["rawtext_list"]
    individuals = clean_individual_format_anon(request.json.get("individuals", {}))

    Pseudonymizer = TextAnonymizer(rawtext_list, individuals=individuals)
    masked_corpus = Pseudonymizer.mask_corpus()

    response = {
        "output": masked_corpus,
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
