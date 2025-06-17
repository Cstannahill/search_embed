import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import random
from collections import defaultdict
import time
from duckduckgo_search import DDGS
import httpx
from enum import Enum
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult:
    """Standardized search result across providers"""

    def __init__(self, url: str, title: str, snippet: str, provider: str):
        self.url = url
        self.title = title
        self.snippet = snippet
        self.provider = provider

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "provider": self.provider,
        }


class ProviderStatus(Enum):
    """Health status for search providers"""

    HEALTHY = "healthy"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    DISABLED = "disabled"


class SearchProvider(ABC):
    """Abstract base class for search providers"""

    def __init__(self, name: str, requests_per_minute: int = 60):
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.request_times: List[float] = []
        self.status = ProviderStatus.HEALTHY
        self.error_count = 0
        self.last_error_time: Optional[float] = None
        self.cooldown_until: Optional[float] = None

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.requests_per_minute:
            return False
        return True

    def _record_request(self):
        """Record a request timestamp"""
        self.request_times.append(time.time())

    def _handle_error(self, error: Exception):
        """Handle provider errors and update status"""
        self.error_count += 1
        self.last_error_time = time.time()

        # Exponential backoff based on error count
        cooldown_seconds = min(300, 10 * (2 ** (self.error_count - 1)))
        self.cooldown_until = time.time() + cooldown_seconds

        if "rate" in str(error).lower() or "429" in str(error):
            self.status = ProviderStatus.RATE_LIMITED
            logger.warning(
                f"{self.name}: Rate limited, cooling down for {cooldown_seconds}s"
            )
        else:
            self.status = ProviderStatus.ERROR
            logger.error(f"{self.name}: Error occurred: {error}")

    def _reset_if_healthy(self):
        """Reset error count if enough time has passed"""
        if self.cooldown_until and time.time() > self.cooldown_until:
            self.status = ProviderStatus.HEALTHY
            self.error_count = max(0, self.error_count - 1)
            self.cooldown_until = None
            logger.info(f"{self.name}: Recovered, status now HEALTHY")

    def is_available(self) -> bool:
        """Check if provider is available for searches"""
        self._reset_if_healthy()

        if self.status != ProviderStatus.HEALTHY:
            return False

        return self._check_rate_limit()

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Perform search - must be implemented by subclasses"""
        pass


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider"""

    def __init__(self):
        super().__init__("DuckDuckGo", requests_per_minute=30)
        self.ddgs = DDGS()

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        if not self.is_available():
            return []

        self._record_request()
        results = []

        try:
            for result in self.ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        url=result.get("href", ""),
                        title=result.get("title", ""),
                        snippet=result.get("body", ""),
                        provider=self.name,
                    )
                )
        except Exception as e:
            self._handle_error(e)
            return []

        return results


class SearxProvider(SearchProvider):
    """Searx/SearxNG search provider"""

    def __init__(self, instance_url: str = "https://searx.be"):
        super().__init__("Searx", requests_per_minute=60)
        self.instance_url = instance_url.rstrip("/")

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Searx instance"""
        if not self.is_available():
            return []

        self._record_request()
        results = []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.instance_url}/search",
                    params={
                        "q": query,
                        "format": "json",
                        "engines": "google,bing,duckduckgo",
                        "pageno": 1,
                    },
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("results", [])[:max_results]:
                        results.append(
                            SearchResult(
                                url=result.get("url", ""),
                                title=result.get("title", ""),
                                snippet=result.get("content", ""),
                                provider=self.name,
                            )
                        )
                else:
                    raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            self._handle_error(e)
            return []

        return results


class BraveSearchProvider(SearchProvider):
    """Brave Search API provider"""

    def __init__(self, api_key: str):
        super().__init__("Brave", requests_per_minute=100)
        self.api_key = api_key
        self.api_url = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search using Brave Search API"""
        if not self.is_available():
            return []

        self._record_request()
        results = []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.api_url,
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                    },
                    params={"q": query, "count": max_results},
                    timeout=10,
                )

                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("web", {}).get("results", []):
                        results.append(
                            SearchResult(
                                url=result.get("url", ""),
                                title=result.get("title", ""),
                                snippet=result.get("description", ""),
                                provider=self.name,
                            )
                        )
                else:
                    raise Exception(f"HTTP {response.status_code}")

        except Exception as e:
            self._handle_error(e)
            return []

        return results


class RotationStrategy(Enum):
    """Search provider rotation strategies"""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED = "weighted"
    LEAST_USED = "least_used"


class SearchProviderManager:
    """Manages multiple search providers with rotation and fallback"""

    def __init__(
        self,
        providers: List[SearchProvider],
        strategy: RotationStrategy = RotationStrategy.ROUND_ROBIN,
    ):
        self.providers = providers
        self.strategy = strategy
        self.current_index = 0
        self.usage_count = defaultdict(int)
        self.success_count = defaultdict(int)
        self.last_used = defaultdict(float)

    def _get_available_providers(self) -> List[SearchProvider]:
        """Get list of currently available providers"""
        return [p for p in self.providers if p.is_available()]

    def _select_provider(
        self, available: List[SearchProvider]
    ) -> Optional[SearchProvider]:
        """Select a provider based on the rotation strategy"""
        if not available:
            return None

        if self.strategy == RotationStrategy.ROUND_ROBIN:
            # Find the next available provider in sequence
            for _ in range(len(self.providers)):
                provider = self.providers[self.current_index % len(self.providers)]
                self.current_index += 1
                if provider in available:
                    return provider
            return available[0]

        elif self.strategy == RotationStrategy.RANDOM:
            return random.choice(available)

        elif self.strategy == RotationStrategy.LEAST_USED:
            # Select the least recently used available provider
            available_sorted = sorted(available, key=lambda p: self.last_used[p.name])
            return available_sorted[0]

        elif self.strategy == RotationStrategy.WEIGHTED:
            # Weight by success rate
            weights = []
            for provider in available:
                total = self.usage_count[provider.name]
                if total == 0:
                    weights.append(1.0)  # Give new providers a chance
                else:
                    success_rate = self.success_count[provider.name] / total
                    weights.append(success_rate)

            return random.choices(available, weights=weights)[0]

    async def search(
        self, query: str, max_results: int = 10, use_multiple: bool = False
    ) -> List[SearchResult]:
        """
        Perform search using available providers

        Args:
            query: Search query
            max_results: Maximum results per provider
            use_multiple: If True, aggregates results from multiple providers
        """
        results = []
        providers_tried = set()

        if use_multiple:
            # Use multiple providers for diverse results
            available = self._get_available_providers()
            tasks = []

            for provider in available[:3]:  # Use up to 3 providers
                self.last_used[provider.name] = time.time()
                self.usage_count[provider.name] += 1
                tasks.append(provider.search(query, max_results))

            if tasks:
                all_results = await asyncio.gather(*tasks, return_exceptions=True)

                for provider, result in zip(available[:3], all_results):
                    if isinstance(result, Exception):
                        logger.error(f"{provider.name} search failed: {result}")
                    else:
                        self.success_count[provider.name] += 1
                        results.extend(result)
        else:
            # Use single provider with fallback
            while len(results) == 0 and len(providers_tried) < len(self.providers):
                available = [
                    p
                    for p in self._get_available_providers()
                    if p.name not in providers_tried
                ]

                provider = self._select_provider(available)
                if not provider:
                    logger.warning("No available search providers")
                    break

                providers_tried.add(provider.name)
                self.last_used[provider.name] = time.time()
                self.usage_count[provider.name] += 1

                try:
                    results = await provider.search(query, max_results)
                    if results:
                        self.success_count[provider.name] += 1
                        logger.info(f"Search successful with {provider.name}")
                except Exception as e:
                    logger.error(f"{provider.name} search failed: {e}")

        # Deduplicate results by URL
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results[:max_results]

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics for all providers"""
        stats = {}
        for provider in self.providers:
            total = self.usage_count[provider.name]
            stats[provider.name] = {
                "status": provider.status.value,
                "total_requests": total,
                "successful_requests": self.success_count[provider.name],
                "success_rate": (
                    self.success_count[provider.name] / total if total > 0 else 0
                ),
                "error_count": provider.error_count,
                "requests_in_last_minute": len(provider.request_times),
            }
        return stats


# Integration with your existing pipeline
class EnhancedKnowledgeAcquisitionPipeline:
    """Modified pipeline using multiple search providers"""

    def __init__(
        self, topic: str, collection_name: str, search_providers: List[SearchProvider]
    ):
        self.topic = topic
        # ... other initialization ...

        # Initialize search manager with providers
        self.search_manager = SearchProviderManager(
            providers=search_providers, strategy=RotationStrategy.LEAST_USED
        )

    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using provider manager"""
        results = await self.search_manager.search(
            query, max_results=10, use_multiple=False  # Set to True for diverse results
        )

        # Convert to expected format
        return [r.to_dict() for r in results]


# Example usage
async def main():
    # Initialize search providers
    providers = [
        DuckDuckGoProvider(),
        SearxProvider("https://searx.be"),  # Public instance
        # Add Brave if you have an API key
        # BraveSearchProvider("your-api-key-here")
    ]

    # Create search manager
    search_manager = SearchProviderManager(
        providers=providers, strategy=RotationStrategy.LEAST_USED
    )

    # Example searches
    queries = [
        "quantum computing applications",
        "quantum error correction methods",
        "quantum supremacy achievements",
    ]

    for query in queries:
        print(f"\nSearching: {query}")
        results = await search_manager.search(query, max_results=5)

        for result in results:
            print(f"  [{result.provider}] {result.title}")
            print(f"    {result.url}")

        # Show provider stats
        stats = search_manager.get_provider_stats()
        print("\nProvider Statistics:")
        for provider, stat in stats.items():
            print(
                f"  {provider}: {stat['status']} - "
                f"{stat['successful_requests']}/{stat['total_requests']} successful"
            )

        # Small delay between searches
        await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())
