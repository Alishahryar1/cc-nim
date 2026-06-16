from api.models.anthropic import Message
from api.rag import RagEngine


def test_rag_retrieval():
    engine = RagEngine()
    history = [
        Message(role="user", content="How do I bake a cake?"),
        Message(role="assistant", content="You need flour, eggs, and sugar."),
        Message(role="user", content="What about a pie?"),
        Message(
            role="assistant", content="For a pie, you need a crust and fruit filling."
        ),
        Message(role="user", content="Is it raining today?"),
        Message(role="assistant", content="I don't have real-time weather access."),
    ]

    # Query about baking should return cake and pie messages
    query = "Tell me more about baking sweets like cakes"
    relevant = engine.retrieve_relevant(query, history, top_k=2)

    contents = [m.content for m in relevant]
    assert any("cake" in str(c).lower() for c in contents)
    assert any("flour" in str(c).lower() for c in contents)


def test_rag_empty_history():
    engine = RagEngine()
    assert engine.retrieve_relevant("test", []) == []


def test_rag_no_query_tokens():
    engine = RagEngine()
    history = [Message(role="user", content="hi")]
    # Should just return what it can
    assert len(engine.retrieve_relevant("!!!", history)) == 1
