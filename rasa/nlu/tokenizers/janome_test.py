from janome.tokenizer import Tokenizer
t = Tokenizer()
for token in t.tokenize(u'仕組みについて説明します'):
  print(token.node.surface)