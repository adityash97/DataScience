import logging

logger = logging.getLogger(__name__)

name = 'web_search'
description = 'Search the web for recent information. Input: {"query": str}.'


def run(payload: dict) -> dict:
    """
    Lightweight web search. Tries duckduckgo_search if installed,
    otherwise returns a deterministic stub so the demo stays self-contained.
    """
    query = (payload or {}).get('query', '').strip()
    if not query:
        return {'tool': name, 'success': False, 'error': 'Empty query.'}

    logger.info('[web_search_tool] Query: %.80s', query)

    try:
        from duckduckgo_search import DDGS  # optional dependency

        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=3))
        results = [
            {
                'title': h.get('title', ''),
                'snippet': h.get('body', ''),
                'url': h.get('href', ''),
            }
            for h in hits
        ]
        return {'tool': name, 'success': True, 'query': query, 'results': results, 'source': 'duckduckgo'}
    except ImportError:
        logger.info('[web_search_tool] duckduckgo_search not installed — returning stub results')
        return {
            'tool': name,
            'success': True,
            'query': query,
            'results': [
                {'title': f'Result 1 for "{query}"', 'snippet': 'Stub snippet — install duckduckgo_search for live results.', 'url': 'https://example.com/1'},
                {'title': f'Result 2 for "{query}"', 'snippet': 'Stub snippet — install duckduckgo_search for live results.', 'url': 'https://example.com/2'},
            ],
            'source': 'stub',
        }
    except Exception as exc:
        logger.exception('[web_search_tool] Search failed')
        return {'tool': name, 'success': False, 'error': str(exc)}
