import spacy
from spacy.tokens import Doc, Token
import spacy_stanza

nlp = spacy_stanza.load_pipeline("en", processors="lemma,tokenize,depparse,pos")


def deompose_verb_root(doc: Doc, root_token: Token):
    rst = []
    # [('obl', 2782), ('nsubj', 2107), ('obj', 1912), ('nsubj:pass', 1434), ('punct', 1202), ('aux:pass', 1108), ('mark', 1085), ('advcl', 553), ('ccomp', 296), ('advmod', 258), ('xcomp', 228), ('obl:tmod', 228), ('aux', 194), ('conj', 135), ('obl:agent', 74), ('parataxis', 64), ('expl', 58), ('compound', 25), ('compound:prt', 22), ('obl:npmod', 20), ('dep', 15), ('dislocated', 8), ('amod', 6), ('cop', 5), ('discourse', 4), ('advcl:relcl', 4), ('det', 3), ('nsubj:outer', 1), ('vocative', 1), ('cc:preconj', 1), ('csubj', 1), ('cc', 1), ('case', 1)]
    noun_tokens = []
    other_dep_tokens = []
    for child in root_token.children:
        if child.pos_ == "NOUN":  # and child.dep_ in ['obl', 'nsubj', 'obj', 'nsubj:pass']:
            noun_tokens.append(child)
        else:
            other_dep_tokens.append(child)

    noun_tokens.sort(key=lambda x: x.idx)
    current_tokens_used = [root_token]
    if noun_tokens:
        for noun_token in noun_tokens:
            current_tokens_used.extend([t for t in noun_token.subtree])
            current_tokens_used.sort(key=lambda x: x.idx)
            verb_noun_sent = " ".join([x.text for x in current_tokens_used])
            rst.append(verb_noun_sent)

    if other_dep_tokens:
        if len(other_dep_tokens) == 1 and other_dep_tokens[0].pos_ == "PUNCT":
            return rst
        rst.append(doc.text)

    return rst


def process_noun_subtree(subtree, root_token):
    rst = []

    conj_tokens = []
    amod_tokens = []
    nmod_tokens = []
    compound_tokens = []
    case_tokens = []
    other_dep_tokens = []
    for child in root_token.children:
        if child.dep_ == 'conj':
            conj_tokens.append(child)
        elif child.dep_ == 'amod':
            amod_tokens.append(child)
        elif child.dep_ == 'case':
            case_tokens.append(child)
        elif child.dep_ == 'nmod':
            nmod_tokens.append(child)
        elif child.dep_ == 'compound':
            compound_tokens.append(child)
        elif child.dep_ == 'det':
            pass
        else:
            other_dep_tokens.append(child)

    current_tokens_used = [root_token]
    if case_tokens:
        for case_token in case_tokens:
            current_tokens_used.extend([t for t in case_token.subtree])

    if compound_tokens:
        for compound_token in compound_tokens:
            current_tokens_used.extend([t for t in compound_token.subtree])

    current_tokens_used.sort(key=lambda x: x.idx)
    # compounds_sent = " ".join([x.text for x in current_tokens_used])
    rst.append(current_tokens_used.copy())

    if conj_tokens:
        for conj_token in conj_tokens:
            current_tokens_used.extend([t for t in conj_token.subtree])
        current_tokens_used.sort(key=lambda x: x.idx)
        # conj_sent = " ".join([x.text for x in current_tokens_used])
        rst.append(current_tokens_used.copy())

    if amod_tokens:
        for amod_token in amod_tokens:
            current_tokens_used.extend([t for t in amod_token.subtree])
        current_tokens_used.sort(key=lambda x: x.idx)
        # amod_sent = " ".join([x.text for x in current_tokens_used])
        rst.append(current_tokens_used.copy())

    if nmod_tokens:
        for nmod_token in nmod_tokens:
            # print(decompose_noun_root(nmod_token.subtree, nmod_token))
            current_tokens_used.extend([t for t in nmod_token.subtree])
        current_tokens_used.sort(key=lambda x: x.idx)
        # nmod_sent = " ".join([x.text for x in current_tokens_used])
        rst.append(current_tokens_used.copy())

    if other_dep_tokens:
        rst.append([x for x in subtree])

    return rst


def decompose_noun_root(doc: Doc, root_token):
    rst = []

    conj_tokens = []
    amod_tokens = []
    nmod_tokens = []
    compound_tokens = []
    other_dep_tokens = []
    for child in root_token.children:
        if child.dep_ == 'conj':
            conj_tokens.append(child)
        elif child.dep_ == 'amod':
            amod_tokens.append(child)
        elif child.dep_ == 'nmod':
            nmod_tokens.append(child)
        elif child.dep_ == 'compound':
            compound_tokens.append(child)
        elif child.dep_ == 'det':
            pass
        else:
            other_dep_tokens.append(child)

    current_tokens_used = [root_token]
    rst.append(root_token.text)

    if compound_tokens:
        for compound_token in compound_tokens:
            current_tokens_used.extend([t for t in compound_token.subtree])
        current_tokens_used.sort(key=lambda x: x.idx)
        compounds_sent = " ".join([x.text for x in current_tokens_used])
        rst.append(compounds_sent)

    if conj_tokens:
        for conj_token in conj_tokens:
            current_tokens_used.extend([t for t in conj_token.subtree])
        current_tokens_used.sort(key=lambda x: x.idx)
        conj_sent = " ".join([x.text for x in current_tokens_used])
        rst.append(conj_sent)

    if amod_tokens:
        for amod_token in amod_tokens:
            current_tokens_used.extend([t for t in amod_token.subtree])
        current_tokens_used.sort(key=lambda x: x.idx)
        amod_sent = " ".join([x.text for x in current_tokens_used])
        rst.append(amod_sent)

    if nmod_tokens:
        for nmod_token in nmod_tokens:
            subs = process_noun_subtree(nmod_token.subtree, nmod_token)
            for sub in subs:
                tokens = current_tokens_used + sub
                tokens.sort(key=lambda x: x.idx)
                nmod_sub_sent = " ".join([x.text for x in tokens])
                rst.append(nmod_sub_sent)

            # print(decompose_noun_root(nmod_token.subtree, nmod_token))
            current_tokens_used.extend(subs[-1])
        # current_tokens_used.sort(key=lambda x:x.idx)
        # nmod_sent = " ".join([x.text for x in current_tokens_used])
        # rst.append(nmod_sent)
    if other_dep_tokens:
        rst.append(doc.text)

    return rst


def decompose_sentence_with_stanza(doc: Doc):
    root_token = None
    for token in doc:
        if token.dep_ == "root":
            root_token = token
            break
    if root_token == None:
        return []

    if root_token.pos_ == 'NOUN':
        return decompose_noun_root(doc, root_token)
    elif root_token.pos_ == "VERB":
        return deompose_verb_root(doc, root_token)
    else:
        return [root_token.text, doc.text]