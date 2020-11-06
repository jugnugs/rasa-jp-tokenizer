import pytest

from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import TEXT, INTENT, ACTION_TEXT, ACTION_NAME
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.tokenizers.janome_tokenizer import JanomeTokenizer

@pytest.mark.parametrize(
    "text, expected_tokens, expected_indices",
    [
        (
            "私が学生です",
            ["私", "が", "学生", "です"],
            [(0, 1), (1, 2), (2, 4), (4, 6)],
        ),
        (
            "５月にアンタがマクドナルドにいるみたい",
            ["５月", "に", "アンタ", "が", "マクドナルド", "に", "いる", "みたい"],
            [(0, 2), (2, 3), (3, 6), (6, 7), (7, 13), (13, 14), (14, 16), (16, 19)],
        ),
        (
            "東京スカイツリーへのお越しは、東武スカイツリーライン「とうきょうスカイツリー駅」が便利です。",
            ["東京", "スカイ", "ツリー", "へ", "の", "お越し", "は", "東武", "スカイツリーライン", "とう", "きょう", "スカイ", "ツリー", "駅", "が", "便利", "です"],
            [(0, 2), (2, 5), (5, 8), (8, 9), (9, 10), (10, 13), (13, 14), (14, 16), (16, 25), (25, 27), (27, 30), (30, 33), (33, 36), (36, 37), (37, 38), (38, 40), (40, 42)],
        )
    ],
)
def test_janome(text, expected_tokens, expected_indices):
    tk = JanomeTokenizer()

    tokens = tk.tokenize(Message(data={TEXT: text}), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]
