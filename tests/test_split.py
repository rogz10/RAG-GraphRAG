from eval.split import split_questions


def test_no_chunk_in_two_splits():
    ids = [f"c{i}" for i in range(100)] * 2# 100 chunks, chacun 2 fois (2 questions)
    splits = split_questions(ids)

    assert not splits["train"] & splits["val"]
    assert not splits["train"] & splits["test"]
    assert not splits["val"] & splits["test"]
    assert len(splits["train"] | splits["val"] | splits["test"]) == 100


def test_same_seed_same_split():
    ids = [f"c{i}" for i in range(100)]
    assert split_questions(ids) == split_questions(ids)
