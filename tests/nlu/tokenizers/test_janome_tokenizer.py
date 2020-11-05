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
            "形態素解析器には色々ありますが、中でもメジャーと思われる MeCab の仕組みについて説明します。",
            ["形態素解析器", "に", "は", "色々", "あり", "ます", "が", "中", "で", "も", "メジャー", "と", "思わ", "れる", "MeCab", "の", "仕組み", "について", "説明", "し", "ます"],
            [(0, 6), (6, 7), (7, 8), (8, 10), (10, 12), (12, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 22), (22, 23), (23, 25), (25, 27), (27, 32), (32, 33), (33, 36), (36, 40), (40, 42), (42, 43), (43, 45)],
        ),
        (
            "５月にアンタがマクドナルドにいるみたい",
            ["５月", "に", "アンタ", "が", "マクドナルド", "に", "いる", "みたい"],
            [(0, 2), (2, 3), (3, 6), (6, 7), (7, 13), (13, 14), (14, 16), (16, 19)],
        )
    ],
)
def test_janome(text, expected_tokens, expected_indices):
    tk = JanomeTokenizer()

    tokens = tk.tokenize(Message(data={TEXT: text}), attribute=TEXT)

    assert [t.text for t in tokens] == expected_tokens
    assert [t.start for t in tokens] == [i[0] for i in expected_indices]
    assert [t.end for t in tokens] == [i[1] for i in expected_indices]
