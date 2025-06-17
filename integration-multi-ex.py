# Integration example - add this to your existing pipeline

from typing import List, Dict, Any
import os


class KnowledgeAcquisitionPipeline:
    def __init__(
        self, topic: str, collection_name: str, use_multi_provider: bool = True
    ):
        self.topic = topic
        self.ollama = AsyncClient()
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"topic": topic}
        )
        self.processed_urls = set()
        self.knowledge_coverage = defaultdict(int)
        self.search_history = []

        # Initialize search system
        if use_multi_provider:
            self._init_multi_provider_search()
        else:
            # Fallback to single provider
            self.search_manager = None

    def _init_multi_provider_search(self):
        """Initialize multiple search providers"""
        providers = []

        # DuckDuckGo (no API key needed)
        providers.append(DuckDuckGoProvider())

        # Searx instances (try multiple for redundancy)
        searx_instances = [
            "https://searx.be",
            "https://searx.info",
            "https://searx.bar",
        ]
        for instance in searx_instances[:2]:  # Use 2 instances
            try:
                providers.append(SearxProvider(instance))
            except Exception as e:
                print(f"Could not add Searx instance {instance}: {e}")

        # Brave Search (if API key available)
        brave_api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if brave_api_key:
            providers.append(BraveSearchProvider(brave_api_key))

        # You can add more providers here:
        # - Bing Search API (requires Azure account)
        # - Google Custom Search (requires API key)
        # - Yandex, Qwant, etc.

        self.search_manager = SearchProviderManager(
            providers=providers, strategy=RotationStrategy.LEAST_USED
        )

        print(f"Initialized {len(providers)} search providers")

    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using appropriate method"""
        if self.search_manager:
            # Use multi-provider search
            results = await self.search_manager.search(
                query,
                max_results=10,
                use_multiple=len(self.search_history) % 5
                == 0,  # Every 5th query uses multiple
            )
            return [r.to_dict() for r in results]
        else:
            # Fallback to original DuckDuckGo implementation
            return await self._search_duckduckgo(query)

    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Original DuckDuckGo search implementation"""
        ddgs = DDGS()
        results = []

        try:
            for result in ddgs.text(query, max_results=10):
                results.append(
                    {
                        "url": result["href"],
                        "title": result["title"],
                        "snippet": result["body"],
                        "provider": "DuckDuckGo",
                    }
                )
        except Exception as e:
            print(f"Search error for '{query}': {e}")

        return results

    async def continuous_acquisition_with_monitoring(self, duration_hours: float = 1.0):
        """Enhanced acquisition with provider monitoring"""
        end_time = datetime.now().timestamp() + (duration_hours * 3600)
        iteration_count = 0

        while datetime.now().timestamp() < end_time:
            iteration_count += 1
            print(f"\n--- Iteration {iteration_count} ---")

            # Show provider health every 5 iterations
            if self.search_manager and iteration_count % 5 == 0:
                stats = self.search_manager.get_provider_stats()
                print("\nProvider Health Check:")
                for name, stat in stats.items():
                    print(
                        f"  {name}: {stat['status']} "
                        f"(success rate: {stat['success_rate']:.2%})"
                    )

            # Regular acquisition process
            queries = await self.generate_search_queries(
                use_adaptive=iteration_count > 2
            )

            for query in queries:
                print(f"Searching: {query}")
                results = await self.search_web(query)

                # Show which provider was used
                if results and "provider" in results[0]:
                    providers_used = set(r.get("provider", "Unknown") for r in results)
                    print(f"  Using providers: {', '.join(providers_used)}")

                # Rest of the processing...
                for result in results:
                    content = await self.extract_content(result["url"])
                    if not content:
                        continue

                    relevance = await self.assess_relevance(content)

                    if relevance > 0.6:
                        print(
                            f"Processing: {result['url']} (relevance: {relevance:.2f})"
                        )

                        chunks = await self.enhanced_content_processor(content)

                        base_metadata = {
                            "url": result["url"],
                            "title": result["title"],
                            "relevance_score": relevance,
                            "timestamp": datetime.now().isoformat(),
                            "search_query": query,
                            "search_provider": result.get("provider", "Unknown"),
                            "iteration": iteration_count,
                        }

                        for chunk in chunks:
                            await self.embed_and_store_chunk(chunk, base_metadata)

                await asyncio.sleep(5)

            await asyncio.sleep(60)


# Complete usage example
async def main_with_providers():
    # Set up environment (optional)
    # os.environ["BRAVE_SEARCH_API_KEY"] = "your-key-here"

    # Create pipeline with multi-provider search
    pipeline = KnowledgeAcquisitionPipeline(
        topic="quantum computing applications",
        collection_name="quantum_computing_kb",
        use_multi_provider=True,
    )

    # Run acquisition with monitoring
    await pipeline.continuous_acquisition_with_monitoring(duration_hours=0.5)

    # Get final statistics
    if pipeline.search_manager:
        print("\nFinal Provider Statistics:")
        stats = pipeline.search_manager.get_provider_stats()
        for name, stat in stats.items():
            print(f"{name}:")
            print(f"  Total requests: {stat['total_requests']}")
            print(f"  Success rate: {stat['success_rate']:.2%}")
            print(f"  Current status: {stat['status']}")

    # Query the knowledge base
    answer = await pipeline.synthesize_knowledge(
        "What are the current challenges in quantum error correction?"
    )
    print(f"\nSynthesized Answer:\n{answer}")


# Configuration for different search strategies
SEARCH_CONFIGS = {
    "conservative": {
        "providers": [DuckDuckGoProvider()],
        "strategy": RotationStrategy.ROUND_ROBIN,
        "rate_multiplier": 0.5,  # Use only 50% of rate limits
    },
    "balanced": {
        "providers": [
            DuckDuckGoProvider(),
            SearxProvider("https://searx.be"),
        ],
        "strategy": RotationStrategy.LEAST_USED,
        "rate_multiplier": 0.8,
    },
    "aggressive": {
        "providers": [
            DuckDuckGoProvider(),
            SearxProvider("https://searx.be"),
            SearxProvider("https://searx.info"),
            # Add more providers
        ],
        "strategy": RotationStrategy.WEIGHTED,
        "rate_multiplier": 1.0,
        "use_multiple": True,  # Search multiple providers simultaneously
    },
}

if __name__ == "__main__":
    asyncio.run(main_with_providers())
