from document_agent.config import settings
from document_agent.domain import Chunk
from document_agent.models import get_embeddings
from document_agent.store.vector import VectorStore


def make_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        index=0,
        text=text,
        metadata={
            "doc_id": doc_id,
            "index": 0,
            "source_uri": f"{doc_id}.txt",
        },
    )


def make_store(tmp_path, monkeypatch) -> VectorStore:
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    return VectorStore(
        embedding_function=get_embeddings(),
    )


def test_search_finds_paraphrased_note_first(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)

    chunks = [
        make_chunk(
            "garden:0",
            "garden",
            "The tomatoes by the back fence need water every morning.",
        ),
        make_chunk(
            "car:0",
            "car",
            "The oil change is due before the car hits 80,000 miles.",
        ),
        make_chunk(
            "trip:0",
            "trip",
            "The hotel in Madison is about ten minutes from the lake.",
        ),
    ]

    store.add(chunks)

    results = store.search(
        "The plants outside should be watered each day",
        k=3,
    )

    assert results[0][0].doc_id == "garden"


def test_delete_doc_removes_only_that_documents_chunks(
    tmp_path,
    monkeypatch,
):
    store = make_store(tmp_path, monkeypatch)

    chunks = [
        make_chunk(
            "recipes:0",
            "recipes",
            "Mom's chili needs another fifteen minutes on low heat.",
        ),
        make_chunk(
            "recipes:1",
            "recipes",
            "Add the shredded cheese after taking the pot off the stove.",
        ),
        make_chunk(
            "chores:0",
            "chores",
            "Take the recycling bin out Wednesday night.",
        ),
    ]

    store.add(chunks)

    assert store.count() == 3

    store.delete_doc("recipes")

    assert store.count() == 1

    results = store._collection.get()

    assert results["ids"] == ["chores:0"]


def test_data_survives_store_restart(tmp_path, monkeypatch):
    store = make_store(tmp_path, monkeypatch)

    chunk = make_chunk(
        "car:0",
        "car",
        "The next oil change should be around 80,000 miles.",
    )

    store.add([chunk])

    assert store.count() == 1

    new_store = make_store(tmp_path, monkeypatch)

    assert new_store.count() == 1

    results = new_store.search(
        "When should I get the oil changed?",
        k=1,
    )

    assert results[0][0].chunk_id == "car:0"


def test_re_adding_same_chunk_id_does_not_create_duplicate(
    tmp_path,
    monkeypatch,
):
    store = make_store(tmp_path, monkeypatch)

    chunk = make_chunk(
        "shopping:0",
        "shopping",
        "Buy coffee, paper towels, and dishwasher detergent.",
    )

    store.add([chunk])
    store.add([chunk])

    assert store.count() == 1


def test_re_adding_same_chunk_id_updates_existing_chunk(
    tmp_path,
    monkeypatch,
):
    store = make_store(tmp_path, monkeypatch)

    original = make_chunk(
        "shopping:0",
        "shopping",
        "Buy coffee and paper towels.",
    )
    updated = make_chunk(
        "shopping:0",
        "shopping",
        "Buy coffee, paper towels, and dishwasher detergent.",
    )

    store.add([original])
    store.add([updated])

    assert store.count() == 1

    results = store._collection.get(ids=["shopping:0"])

    assert results["ids"] == ["shopping:0"]
    assert results["documents"] == [
        "Buy coffee, paper towels, and dishwasher detergent."
    ]