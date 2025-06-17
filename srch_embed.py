import asyncio
import aiohttp
from typing import List, Dict, Any
import chromadb
from ollama import AsyncClient
import hashlib
from datetime import datetime
import json
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from collections import defaultdict
import random
import os

# Import search provider utilities
from multi import (
    DuckDuckGoProvider,
    SearxProvider,
    BraveSearchProvider,
    SearchProviderManager,
    RotationStrategy,
)


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
        self.search_manager = None

        if use_multi_provider:
            self._init_multi_provider_search()

    def _init_multi_provider_search(self):
        """Initialize multiple search providers"""
        providers = [DuckDuckGoProvider()]

        # Add a couple of public Searx instances
        for instance in ["https://searx.be", "https://searx.info"]:
            try:
                providers.append(SearxProvider(instance))
            except Exception as e:
                print(f"Could not add Searx instance {instance}: {e}")

        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if brave_key:
            providers.append(BraveSearchProvider(brave_key))

        self.search_manager = SearchProviderManager(
            providers=providers, strategy=RotationStrategy.LEAST_USED
        )

    async def generate_search_queries(self, use_adaptive: bool = True) -> List[str]:
        """Generate search queries, optionally using adaptive strategy"""
        if use_adaptive and len(self.search_history) > 10:
            # Use adaptive search after initial knowledge gathering
            return await self.adaptive_search()
        else:
            # Use standard diverse query generation
            return await self._generate_diverse_queries()

    async def _generate_diverse_queries(self) -> List[str]:
        """Standard diverse query generation"""
        prompt = f"""Generate 5 diverse search queries to comprehensively explore the topic: {self.topic}
        
        Consider different angles:
        - Technical specifications
        - Recent developments
        - Practical applications
        - Research papers
        - Industry perspectives
        
        Return only the queries, one per line."""

        response = await self.ollama.chat(
            model="deepseek-r1",
            messages=[{"role": "user", "content": prompt}],
        )

        queries = response["message"]["content"].strip().split("\n")
        queries = [q.strip() for q in queries if q.strip()]
        self.search_history.extend(queries)
        return queries

    async def adaptive_search(self) -> List[str]:
        """Adjust search strategy based on knowledge gaps"""
        # Get existing knowledge summary
        try:
            current_docs = self.collection.get(limit=50)

            if not current_docs["metadatas"]:
                # Fallback to diverse queries if no existing knowledge
                return await self._generate_diverse_queries()

            # Extract titles and topics covered
            covered_topics = [m.get("title", "") for m in current_docs["metadatas"]]
            covered_urls = [m.get("url", "") for m in current_docs["metadatas"]]

            # Analyze coverage
            prompt = f"""Analyze our knowledge coverage for '{self.topic}'.
            
            Topics already covered:
            {json.dumps(covered_topics[:20], indent=2)}
            
            Previous searches:
            {json.dumps(self.search_history[-10:], indent=2)}
            
            Identify 3-5 critical knowledge gaps and generate targeted search queries to fill them.
            Consider:
            - What subtopics are missing?
            - What recent developments might we have missed?
            - What technical details need deeper exploration?
            - What alternative perspectives haven't been covered?
            
            Return only the search queries, one per line."""

            response = await self.ollama.chat(
                model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
            )

            queries = response["message"]["content"].strip().split("\n")
            queries = [q.strip() for q in queries if q.strip()]

            # Add variation to prevent search loops
            queries = self._add_query_variations(queries)

            self.search_history.extend(queries)
            return queries[:5]  # Limit to 5 queries

        except Exception as e:
            print(f"Adaptive search error: {e}")
            return await self._generate_diverse_queries()

    def _add_query_variations(self, queries: List[str]) -> List[str]:
        """Add variations to queries to prevent repetition"""
        variations = []
        time_modifiers = ["latest", "2024", "recent", "current", "new"]

        for query in queries:
            # Add time modifier if not present
            if not any(mod in query.lower() for mod in time_modifiers):
                variations.append(f"{random.choice(time_modifiers)} {query}")
            else:
                variations.append(query)

        return variations

    async def enhanced_content_processor(self, content: str) -> List[Dict[str, Any]]:
        """Chunk content intelligently for better embedding quality"""
        if len(content) < 500:
            # Too short for intelligent chunking
            return [{"text": content, "type": "full"}]

        try:
            prompt = f"""Extract the main concepts and facts from this text as separate, self-contained statements.
            Each statement should be a complete thought that can stand alone.
            Group related statements together.
            
            Text:
            {content[:3000]}
            
            Format as:
            CONCEPT: [concept name]
            - Statement 1
            - Statement 2
            
            CONCEPT: [next concept]
            - Statement 1
            - Statement 2"""

            response = await self.ollama.chat(
                model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response into structured chunks
            chunks = []
            current_concept = None
            current_statements = []

            for line in response["message"]["content"].split("\n"):
                line = line.strip()
                if line.startswith("CONCEPT:"):
                    # Save previous concept if exists
                    if current_concept and current_statements:
                        chunks.append(
                            {
                                "text": f"{current_concept}\n"
                                + "\n".join(current_statements),
                                "type": "concept",
                                "concept": current_concept,
                            }
                        )
                    current_concept = line.replace("CONCEPT:", "").strip()
                    current_statements = []
                elif line.startswith("-") and current_concept:
                    current_statements.append(line)

            # Save last concept
            if current_concept and current_statements:
                chunks.append(
                    {
                        "text": f"{current_concept}\n" + "\n".join(current_statements),
                        "type": "concept",
                        "concept": current_concept,
                    }
                )

            # If no chunks were created, fall back to simple chunking
            if not chunks:
                chunks = [{"text": content[:1500], "type": "full"}]

            return chunks

        except Exception as e:
            print(f"Enhanced processing error: {e}")
            # Fallback to simple chunking
            return [{"text": content[:1500], "type": "full"}]

    async def search_web(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using available providers"""
        if self.search_manager:
            results = await self.search_manager.search(
                query,
                max_results=10,
                use_multiple=len(self.search_history) % 5 == 0,
            )
            return [r.to_dict() for r in results]

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

    async def extract_content(self, url: str) -> str:
        """Extract clean text content from URL"""
        if url in self.processed_urls:
            return ""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, "html.parser")

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (
                            phrase.strip()
                            for line in lines
                            for phrase in line.split("  ")
                        )
                        text = " ".join(chunk for chunk in chunks if chunk)

                        self.processed_urls.add(url)
                        return text[:5000]  # Limit content length

        except Exception as e:
            print(f"Content extraction error for {url}: {e}")

        return ""

    async def assess_relevance(self, content: str) -> float:
        """Use LLM to assess content relevance to topic"""
        prompt = f"""Rate the relevance of this content to the topic '{self.topic}' on a scale of 0-10.
        
        Content excerpt: {content[:500]}...
        
        Respond with only a number between 0 and 10."""

        response = await self.ollama.chat(
            model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
        )

        try:
            score = float(response["message"]["content"].strip())
            return min(10, max(0, score)) / 10
        except:
            return 0.5

    async def embed_and_store_chunk(
        self, chunk: Dict[str, Any], base_metadata: Dict[str, Any]
    ):
        """Generate embeddings and store a single chunk in vector DB"""
        # Generate embedding using nomic-embed-text
        embedding_response = await self.ollama.embeddings(
            model="nomic-embed-text", prompt=chunk["text"]
        )

        # Create unique ID for the chunk
        doc_id = hashlib.md5(chunk["text"].encode()).hexdigest()

        # Enhance metadata with chunk-specific info
        metadata = {**base_metadata}
        metadata["chunk_type"] = chunk["type"]
        if "concept" in chunk:
            metadata["concept"] = chunk["concept"]
            # Track concept coverage
            self.knowledge_coverage[chunk["concept"]] += 1

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding_response["embedding"]],
            documents=[chunk["text"]],
            metadatas=[metadata],
            ids=[doc_id],
        )

    async def continuous_acquisition(
        self, duration_hours: float = 1.0, use_adaptive: bool = True
    ):
        """Main loop for continuous knowledge acquisition with enhancements"""
        end_time = datetime.now().timestamp() + (duration_hours * 3600)
        iteration_count = 0

        while datetime.now().timestamp() < end_time:
            iteration_count += 1
            print(f"\n--- Iteration {iteration_count} ---")

            # Generate search queries (adaptive after initial phase)
            queries = await self.generate_search_queries(
                use_adaptive=use_adaptive and iteration_count > 2
            )

            for query in queries:
                print(f"Searching: {query}")
                results = await self.search_web(query)

                for result in results:
                    # Extract content
                    content = await self.extract_content(result["url"])
                    if not content:
                        continue

                    # Assess relevance
                    relevance = await self.assess_relevance(content)

                    if relevance > 0.6:  # Threshold for inclusion
                        print(
                            f"Processing: {result['url']} (relevance: {relevance:.2f})"
                        )

                        # Use enhanced content processor for intelligent chunking
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

                        # Store each chunk separately with enhanced metadata
                        for chunk in chunks:
                            await self.embed_and_store_chunk(chunk, base_metadata)

                # Rate limiting
                await asyncio.sleep(5)

            # Log knowledge coverage
            if iteration_count % 3 == 0:
                print(f"\nKnowledge coverage summary:")
                for concept, count in sorted(
                    self.knowledge_coverage.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    print(f"  {concept}: {count} chunks")

            # Wait before next iteration
            await asyncio.sleep(60)

    async def synthesize_knowledge(self, question: str) -> str:
        """Query the vector DB and synthesize an answer"""
        # Search vector DB
        results = self.collection.query(query_texts=[question], n_results=8)

        if not results or not results.get("documents") or not results["documents"][0]:
            return "No relevant information found in the knowledge base."

        # Group by concept if available
        concept_groups = defaultdict(list)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):
            concept = meta.get("concept", "general")
            concept_groups[concept].append(doc)

        # Prepare structured context
        context_parts = []
        for concept, docs in concept_groups.items():
            if concept != "general":
                context_parts.append(f"### {concept}")
            context_parts.extend(docs)

        context = "\n\n".join(context_parts)

        # Generate synthesis using LLM
        prompt = f"""Based on the following information organized by concepts, provide a comprehensive answer to: {question}
        
        Context:
        {context}
        
        Synthesize the information, highlighting connections between concepts and providing a clear, structured response."""

        response = await self.ollama.chat(
            model="deepseek-r1", messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        collection_data = self.collection.get()

        stats = {
            "total_chunks": len(collection_data["ids"]),
            "unique_sources": len(self.processed_urls),
            "concepts_covered": dict(self.knowledge_coverage),
            "search_queries_used": len(self.search_history),
            "latest_searches": self.search_history[-5:],
        }

        return stats


# Enhanced usage example
async def main():
    pipeline = KnowledgeAcquisitionPipeline(
        topic="quantum computing applications", collection_name="quantum_computing_kb"
    )

    # Run continuous acquisition with adaptive search
    await pipeline.continuous_acquisition(duration_hours=0.5, use_adaptive=True)

    # Get knowledge base statistics
    stats = pipeline.get_knowledge_stats()
    print(f"\nKnowledge Base Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Unique sources: {stats['unique_sources']}")
    print(f"Concepts covered: {len(stats['concepts_covered'])}")

    # Query the knowledge base
    answer = await pipeline.synthesize_knowledge(
        "What are the current challenges in quantum error correction?"
    )
    print(f"\nSynthesized Answer:\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
