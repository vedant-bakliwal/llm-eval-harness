from evalharness.corpus import load_corpus, slugify
from evalharness.retriever import BM25Retriever, tokenize

from helpers import CORPUS


def test_corpus_is_split_into_citable_sections():
    chunks = load_corpus(CORPUS)
    ids = [c.id for c in chunks]
    assert "pto#carryover" in ids and "vendors#supplier-notes" in ids
    assert len(ids) == len(set(ids))
    assert slugify("Multi-factor authentication") == "multi-factor-authentication"


def test_tokens_keep_amounts_and_drop_stopwords():
    assert tokenize("What is the $0.67 rate?") == ["$0.67", "rate"]


def test_bm25_ranks_the_right_section_first():
    retriever = BM25Retriever(load_corpus(CORPUS))
    assert retriever.search("mileage reimbursement for my car")[0].chunk.id == "expenses#mileage"
    assert retriever.search("parental leave weeks")[0].chunk.id == "benefits#parental-leave"


def test_unrelated_question_finds_nothing():
    assert BM25Retriever(load_corpus(CORPUS)).search("football world cup winner") == []
