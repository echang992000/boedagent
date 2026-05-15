"""Tests for the literature API clients.

Every client is tested against a mock ``fetcher`` so the suite is
offline and deterministic (no VCR cassettes required).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from boed_agent.literature.clients.arxiv import ArxivClient
from boed_agent.literature.clients.base import ClientConfig
from boed_agent.literature.clients.openalex import OpenAlexClient
from boed_agent.literature.clients.pubmed import PubMedClient
from boed_agent.literature.clients.semantic_scholar import SemanticScholarClient


def test_semantic_scholar_parses_payload():
    payload = json.dumps(
        {
            "data": [
                {
                    "title": "Paper X",
                    "abstract": "About BOED.",
                    "authors": [{"name": "Ada"}],
                    "year": 2022,
                    "citationCount": 5,
                    "externalIds": {"DOI": "10.0/x", "ArXiv": "2201.00001"},
                    "url": "https://example.org/paper",
                }
            ]
        }
    )
    client = SemanticScholarClient(
        config=ClientConfig(
            rate_limit_interval=0.0, fetcher=lambda url, hdr: payload
        )
    )
    results = client.search("boed", limit=1)
    assert len(results) == 1
    assert results[0].doi == "10.0/x"
    assert results[0].arxiv_id == "2201.00001"
    assert results[0].citation_count == 5


def test_arxiv_parses_atom_feed():
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <title>Optimal Design</title>
        <summary>We propose an EIG estimator.</summary>
        <published>2021-05-12T00:00:00Z</published>
        <id>http://arxiv.org/abs/2105.01234</id>
        <author><name>Ada</name></author>
        <link type="text/html" href="https://arxiv.org/abs/2105.01234"/>
      </entry>
    </feed>
    """
    client = ArxivClient(
        config=ClientConfig(rate_limit_interval=0.0, fetcher=lambda url, hdr: atom)
    )
    results = client.search("optimal design", limit=1)
    assert len(results) == 1
    assert results[0].arxiv_id == "2105.01234"
    assert results[0].year == 2021


def test_openalex_inverts_abstract_index():
    body = json.dumps(
        {
            "results": [
                {
                    "title": "A Work",
                    "ids": {"doi": "https://doi.org/10.0/y"},
                    "abstract_inverted_index": {
                        "Hello": [0],
                        "world": [1],
                    },
                    "publication_year": 2020,
                    "cited_by_count": 3,
                    "authorships": [{"author": {"display_name": "Ada"}}],
                    "id": "https://openalex.org/W1",
                }
            ]
        }
    )
    client = OpenAlexClient(
        config=ClientConfig(rate_limit_interval=0.0, fetcher=lambda url, hdr: body)
    )
    results = client.search("hello", limit=1)
    assert results[0].abstract == "Hello world"
    assert results[0].doi == "10.0/y"


def test_pubmed_handles_empty():
    client = PubMedClient(
        config=ClientConfig(rate_limit_interval=0.0, fetcher=lambda url, hdr: '{"esearchresult": {"idlist": []}}')
    )
    assert client.search("nothing") == []
