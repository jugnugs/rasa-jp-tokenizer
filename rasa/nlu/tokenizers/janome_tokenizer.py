from typing import Any, Dict, List, Text

import regex
import re

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message


class JanomeTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": False,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super().__init__(component_config)

    @classmethod
    def required_packages(cls) -> List[Text]:
      return ["janome"]

    def removePunctuation(self, text: Text) -> Text:
      punctuation = "[｛｝（）［］【】、…。〜「」『』：！？]"
      return re.sub(punctuation, "",  text)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        from janome.tokenizer import Tokenizer
        text = message.get(attribute)
        text = self.removePunctuation(text)

        tokenizer = Tokenizer()
        tokenized = tokenizer.tokenize(text)
        tokens = []
        for token in tokenized:
          tokens.append(Token(token.node.surface, token.node.pos - 1))

        return self._apply_token_pattern(tokens)

