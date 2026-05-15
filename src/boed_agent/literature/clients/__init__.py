"""HTTP clients for the literature search sources.

Each client is intentionally small.  Network access is isolated to
:meth:`search` so that tests can monkeypatch it with a pure-Python
responder.  Rate limiting and retries rely on ``tenacity`` when it is
installed; otherwise the clients fall back to a plain loop with
exponential backoff implemented locally.
"""

from boed_agent.literature.clients.arxiv import ArxivClient
from boed_agent.literature.clients.local_corpus import LocalCorpusClient
from boed_agent.literature.clients.openalex import OpenAlexClient
from boed_agent.literature.clients.pubmed import PubMedClient
from boed_agent.literature.clients.semantic_scholar import SemanticScholarClient
from boed_agent.literature.clients.unpaywall import UnpaywallClient

__all__ = [
    "ArxivClient",
    "LocalCorpusClient",
    "OpenAlexClient",
    "PubMedClient",
    "SemanticScholarClient",
    "UnpaywallClient",
]
