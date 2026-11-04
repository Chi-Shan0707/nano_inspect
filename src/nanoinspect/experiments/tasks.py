"""Synthetic prompts with verified single-token alignment.

Every example is tokenized once as full text; the character spans of answer
values and query keys are mapped to token positions with offset mappings and
checked to be exactly one expected token, so clean/corrupted pairs differ at
one known position and per-example positions are aligned for patching.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

KEYS = ("red blue green yellow black white pink purple orange brown gray silver gold "
        "cat dog horse cow pig fox wolf bear lion").split()
VALUES = ("apple coin ring book pen cup ball shoe hat map card bell stone lamp key fork "
          "chair desk bottle knife spoon rope").split()
FILLER = ["The weather was calm that afternoon.", "A train passed slowly through the valley.",
          "Someone was reading a newspaper nearby.", "The river flowed quietly past the town.",
          "Birds were singing in the tall trees.", "A small boat drifted across the lake.",
          "The market opened early in the morning.", "Children played games in the park."]


@dataclass
class RetrievalExample:
    text: str
    ids: list
    keys: list
    values: list
    query: int
    target: str
    target_id: int
    value_ids: list
    positions: dict
    condition: dict = field(default_factory=dict)

    @property
    def distance(self):
        """Tokens from the queried value to the final (prediction) position."""
        return self.positions["final"] - self.positions["value"]


def _render(fmt, keys, values, query, filler):
    """Return text and character spans of each value and of the final query key."""
    parts, spans = [], []
    def add(text, value_index=None):
        start = sum(len(p) for p in parts)
        parts.append(text)
        return start
    fill = " ".join(FILLER[i % len(FILLER)] for i in range(filler))
    if fmt == "sentence":
        add("Facts:\n")
        for i, (k, v) in enumerate(zip(keys, values)):
            start = add(f"The {k} box contains the ")
            start = add(v)
            spans.append((start, start + len(v)))
            add(".\n")
        if filler:
            add(fill + "\n")
        control_start = add("Question")
        add(f": What does the {keys[query]} box contain?\nAnswer: The ")
        key_start = add(keys[query])
        add(" box contains the")
        control = (control_start, control_start + len("Question"))
    elif fmt == "list":
        for i, (k, v) in enumerate(zip(keys, values)):
            add(f"{k}: ")
            start = add(v)
            spans.append((start, start + len(v)))
            add("\n")
        if filler:
            add(fill + "\n")
        key_start = add(keys[query])
        add(":")
        control = (0, len(keys[0]))  # first key; the list format has no question line
    else:
        raise ValueError(f"Unknown retrieval format {fmt!r}")
    return "".join(parts), spans, (key_start, key_start + len(keys[query])), control


def _token_at(offsets, span):
    """Index of the single token whose offsets end at the span end and cover its start."""
    hits = [i for i, (s, e) in enumerate(offsets) if e == span[1] and s <= span[0]]
    return hits[0] if len(hits) == 1 else None


def build_example(tokenizer, fmt, keys, values, query, filler, condition=None):
    text, spans, key_span, control_span = _render(fmt, keys, values, query, filler)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    value_ids = [tokenizer.encode(" " + v, add_special_tokens=False) for v in values]
    if any(len(v) != 1 for v in value_ids):
        raise ValueError("Values must be single tokens with a leading space")
    value_ids = [v[0] for v in value_ids]
    positions = [_token_at(offsets, span) for span in spans]
    if any(p is None for p in positions) or any(ids[p] != v for p, v in zip(positions, value_ids)):
        return None  # tokenization merged a value with its context; reject rather than misalign
    key_position, control_position = _token_at(offsets, key_span), _token_at(offsets, control_span)
    if key_position is None or control_position is None:
        return None
    return RetrievalExample(text, ids, list(keys), list(values), query, values[query], value_ids[query], value_ids,
                            {"value": positions[query], "query_key": key_position, "final": len(ids) - 1,
                             "control": control_position, "fact_values": positions},
                            dict(condition or {}))


def retrieval_examples(tokenizer, fmt, n_facts, filler, count, seed, split):
    """Deterministic examples for one condition; ``split`` separates pilot and held-out streams."""
    if n_facts > len(KEYS) or n_facts + 1 > len(VALUES):
        raise ValueError("Not enough distinct keys/values for this condition")
    rng = random.Random(f"{seed}:{split}:{fmt}:{n_facts}:{filler}")
    condition = {"format": fmt, "n_facts": n_facts, "filler_sentences": filler, "split": split}
    examples, attempts = [], 0
    while len(examples) < count:
        attempts += 1
        if attempts > count * 20:
            raise RuntimeError(f"Could not build aligned examples for {condition}")
        keys, values = rng.sample(KEYS, n_facts), rng.sample(VALUES, n_facts)
        example = build_example(tokenizer, fmt, keys, values, rng.randrange(n_facts), filler, condition)
        if example is not None:
            examples.append(example)
    return examples


def corrupt(tokenizer, example, rng):
    """Replace the queried value by an absent value; the token sequence differs at exactly one position."""
    choices = [v for v in VALUES if v not in example.values]
    replacement = rng.choice(choices)
    values = list(example.values)
    values[example.query] = replacement
    corrupted = build_example(tokenizer, example.condition["format"], example.keys, values, example.query,
                              example.condition["filler_sentences"], example.condition)
    if corrupted is None or len(corrupted.ids) != len(example.ids):
        return None
    differing = [i for i, (a, b) in enumerate(zip(example.ids, corrupted.ids)) if a != b]
    if differing != [example.positions["value"]] or corrupted.positions != example.positions:
        return None
    return corrupted


# --- Steering (Study C) -----------------------------------------------------------------------------

POSITIVE_TRAIN = ["wonderful", "amazing", "lovely", "delightful"]
NEGATIVE_TRAIN = ["terrible", "horrible", "dreadful", "miserable"]
POSITIVE_EVAL = [" good", " great", " excellent", " fantastic", " perfect", " nice"]
NEGATIVE_EVAL = [" bad", " poor", " boring", " disappointing", " mediocre", " awful"]
TRAIN_NOUNS = "movie meal hotel concert book game trip show class party flight service".split()
PILOT_NOUNS = "phone laptop camera museum beach pizza burger coffee jacket sofa guitar album novel painting lecture store airport train garden gym".split()
HELDOUT_NOUNS = ("car bike bus taxi ship boat house room kitchen bathroom bed desk lamp clock watch radio "
                 "song music film play opera ballet dance poem story magazine newspaper website app software "
                 "course teacher doctor dentist nurse lawyer bank shop market mall restaurant cafe bar pub "
                 "bakery salad sandwich soup cake bread cheese wine beer tea juice chocolate dinner lunch breakfast "
                 "festival wedding conference meeting interview").split()
TRAIN_TEMPLATE = "I thought the {noun} was {adjective}."
EVAL_TEMPLATE = "Here is my honest review of the {noun}. In short, it was"
JUDGE_INSTRUCTION = "Review: {text}\nIs the sentiment of this review positive or negative? Answer with one word."
JUDGE_ANSWERS = ("Positive", "Negative")  # first assistant token, compared as a logit difference
POSITIVE_LEXICON = set("good great excellent fantastic perfect nice wonderful amazing lovely delightful enjoyable "
                       "pleasant beautiful brilliant superb best love loved happy fun impressive awesome recommend".split())
NEGATIVE_LEXICON = set("bad poor boring disappointing mediocre awful terrible horrible dreadful miserable worst "
                       "hate hated sad annoying ugly waste broken rude dull unpleasant disappointed".split())
NEUTRAL_TEXTS = [
    "The library opens at nine in the morning and closes at six in the evening.",
    "Water boils at one hundred degrees Celsius at sea level.",
    "The committee will meet on Tuesday to review the annual budget.",
    "Most trains to the airport leave from the second platform.",
    "The museum exhibits paintings from the seventeenth and eighteenth centuries.",
    "Photosynthesis converts light energy into chemical energy in plants.",
    "The report lists the population of each district in the region.",
    "Please submit the form before the end of the month.",
    "The river runs through three countries before reaching the sea.",
    "A standard deck of playing cards contains fifty-two cards.",
    "The software update changes the layout of the settings menu.",
    "The bridge was built in the late nineteenth century.",
]
CAPITALS = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Japan", "Tokyo"),
            ("Russia", "Moscow"), ("China", "Beijing"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Austria", "Vienna"),
            ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"), ("Norway", "Oslo"),
            ("Sweden", "Stockholm"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"), ("Hungary", "Budapest"),
            ("the Czech Republic", "Prague"), ("Belgium", "Brussels"), ("the Netherlands", "Amsterdam"),
            ("Canada", "Ottawa"), ("Argentina", "Buenos"), ("Peru", "Lima"), ("Chile", "Santiago"),
            ("Colombia", "Bogot"), ("Cuba", "Havana"), ("India", "New"), ("Pakistan", "Islamabad"),
            ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("the Philippines", "Manila"),
            ("South Korea", "Seoul"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Ethiopia", "Addis"),
            ("Morocco", "Rabat"), ("Turkey", "Ankara"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Syria", "Damascus"),
            ("Lebanon", "Beirut"), ("Saudi Arabia", "Riyadh"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
            ("Ukraine", "Kyiv"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"),
            ("Croatia", "Zagreb"), ("Iceland", "Reykjavik"), ("Venezuela", "Caracas"), ("Afghanistan", "Kabul"),
            ("Nepal", "Kathmandu"), ("Bangladesh", "Dhaka"), ("Malaysia", "Kuala"), ("Qatar", "Doha"),
            ("Jordan", "Amman"), ("Libya", "Tripoli"), ("Sudan", "Khartoum"), ("Ghana", "Accra"), ("Senegal", "Dakar"),
            ("Uganda", "Kampala"), ("Zimbabwe", "Harare"), ("Mexico", "Mexico"), ("Brazil", "Bras"),
            ("Switzerland", "Bern"), ("the United Kingdom", "London"), ("Scotland", "Edinburgh"),
            ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Estonia", "Tallinn"), ("Latvia", "Riga"),
            ("Lithuania", "Vilnius"), ("Belarus", "Minsk"), ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"),
            ("Azerbaijan", "Baku"), ("Uzbekistan", "Tashkent"), ("Mongolia", "Ulaanbaatar"), ("Laos", "Vientiane"),
            ("Tunisia", "Tunis"), ("Algeria", "Algiers"), ("Angola", "Luanda"), ("Rwanda", "Kigali"),
            ("Somalia", "Mogadishu"), ("Ecuador", "Quito"), ("Uruguay", "Montevideo"), ("Paraguay", "Asunci"),
            ("Jamaica", "Kingston"), ("Malta", "Valletta"), ("Cyprus", "Nicosia"), ("Albania", "Tirana"),
            ("North Macedonia", "Skopje"), ("Bosnia and Herzegovina", "Sarajevo"), ("Montenegro", "Podgorica"),
            ("Yemen", "Sanaa"), ("Oman", "Muscat"), ("Bahrain", "Manama"), ("North Korea", "Pyongyang"),
            ("Taiwan", "Taipei"), ("Sri Lanka", "Colombo"), ("Cambodia", "Phnom"), ("Tanzania", "Dodoma"),
            ("Luxembourg", "Luxembourg"), ("Kuwait", "Kuwait"), ("Panama", "Panama")]
CAPITAL_TEMPLATE = "Q: What is the capital of {country}?\nA: The capital of {country} is"


def single_token(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not a single token: {ids}")
    return ids[0]


def steering_pairs(tokenizer):
    """Aligned positive/negative training sentences differing only in one adjective token."""
    pairs = []
    for noun in TRAIN_NOUNS:
        for positive, negative in zip(POSITIVE_TRAIN[:2], NEGATIVE_TRAIN[:2]):
            a = tokenizer.encode(TRAIN_TEMPLATE.format(noun=noun, adjective=positive), add_special_tokens=False)
            b = tokenizer.encode(TRAIN_TEMPLATE.format(noun=noun, adjective=negative), add_special_tokens=False)
            if len(a) != len(b) or sum(x != y for x, y in zip(a, b)) != 1:
                raise ValueError(f"Training pair for {noun!r} is not aligned to one token")
            pairs.append({"noun": noun, "positive": positive, "negative": negative, "positive_ids": a, "negative_ids": b})
    return pairs


def steering_prompts(tokenizer, nouns, limit=None):
    """Evaluation prompts; nouns that are not single tokens are skipped so lengths stay equal."""
    prompts = []
    for noun in nouns:
        if len(tokenizer.encode(" " + noun, add_special_tokens=False)) != 1:
            continue
        text = EVAL_TEMPLATE.format(noun=noun)
        prompts.append({"noun": noun, "text": text, "ids": tokenizer.encode(text, add_special_tokens=False)})
        if limit and len(prompts) == limit:
            break
    return prompts


def capital_prompts(tokenizer):
    prompts = []
    for country, capital in CAPITALS:
        text = CAPITAL_TEMPLATE.format(country=country)
        target = tokenizer.encode(" " + capital, add_special_tokens=False)[0]
        prompts.append({"country": country, "answer": capital, "text": text, "target_id": target,
                        "ids": tokenizer.encode(text, add_special_tokens=False)})
    return prompts


def check_disjoint():
    """Training, pilot and held-out nouns, and training vs evaluation adjectives, must not overlap."""
    nouns = [set(TRAIN_NOUNS), set(PILOT_NOUNS), set(HELDOUT_NOUNS)]
    if any(a & b for i, a in enumerate(nouns) for b in nouns[i + 1:]):
        raise ValueError("Steering splits share nouns")
    train = {w.strip() for w in POSITIVE_TRAIN + NEGATIVE_TRAIN}
    evaluation = {w.strip() for w in POSITIVE_EVAL + NEGATIVE_EVAL}
    if train & evaluation:
        raise ValueError(f"Training and evaluation adjectives overlap: {train & evaluation}")
