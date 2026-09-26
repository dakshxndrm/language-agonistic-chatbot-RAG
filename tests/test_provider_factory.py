"""Provider selection (C1) must switch on the env var alone, with no network call
needed to prove it — get_embedder()/get_reranker() should never touch the wire
just to pick a class.
"""

import server.providers.factory as factory
from server.config import Settings
from server.providers.embed_cohere import CohereEmbedder
from server.providers.embed_local import LocalEmbedder
from server.providers.rerank_cohere import CohereReranker
from server.providers.rerank_local import LocalReranker


def reset(monkeypatch, **kw):
    factory._embedder = None
    factory._reranker = None
    settings = Settings(_env_file=None, **kw)
    monkeypatch.setattr(factory, "get_settings", lambda: settings)


def test_cohere_is_the_default(monkeypatch):
    reset(monkeypatch)
    assert isinstance(factory.get_embedder(), CohereEmbedder)
    assert isinstance(factory.get_reranker(), CohereReranker)


def test_onprem_forces_local(monkeypatch):
    reset(monkeypatch, bhasha_profile="onprem")
    assert isinstance(factory.get_embedder(), LocalEmbedder)
    assert isinstance(factory.get_reranker(), LocalReranker)
