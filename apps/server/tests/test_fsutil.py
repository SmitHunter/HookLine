from hookline.fsutil import atomic_write_text, slugify


def test_slugify():
    assert slugify("She said WHAT?! 😱") == "she-said-what"
    assert slugify("😱😱") == "clip"


def test_atomic_write_leaves_no_tmp(tmp_path):
    atomic_write_text(tmp_path / "a.json", "{}")
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]
