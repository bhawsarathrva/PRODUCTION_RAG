import logfire
from flashrank import Ranker, RerankRequest

_ranker: Ranker | None = None


def _get_ranker() -> Ranker:
    """Lazily initialise the local cross-encoder reranker once per process."""
    global _ranker
    if _ranker is None:
        logfire.info("Loading FlashRank cross-encoder reranker.")
        _ranker = Ranker()
    return _ranker


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Reranks candidate document chunks against the query using a local
    cross-encoder and returns the top_n chunk texts, most relevant first.
    """
    if not documents:
        return []

    ranker = _get_ranker()
    passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
    request = RerankRequest(query=query, passages=passages)

    try:
        results = ranker.rerank(request)
    except Exception as e:
        logfire.error(f"❌ Reranking failed, falling back to original order: {e}")
        return documents[:top_n]

    return [result["text"] for result in results[:top_n]]
