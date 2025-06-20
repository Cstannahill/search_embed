"""
Practical Implementation: Optimized srch_embed.py for GTX 1070

This is a drop-in replacement that applies the key optimizations
while maintaining compatibility with your existing code.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
import chromadb
from ollama import AsyncClient
import hashlib
from datetime import datetime
import json
from bs4 import BeautifulSoup
from collections import defaultdict
import random
import os
import time
import numpy as np
import argparse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import chardet
import fitz  # PyMuPDF
import openai  # Ensure this is at the top-level imports
import logging

# --- Logging Configuration (Python 3.13+ best practices) ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Load environment variables early ---
load_dotenv()

# --- Import search provider utilities ---
from multi import (
    DuckDuckGoProvider,
    SearxProvider,
    BraveSearchProvider,
    SearchProviderManager,
    RotationStrategy,
)

# --- GPU optimization imports (with import guard) ---
try:
    import torch
    from sentence_transformers import SentenceTransformer
    GPU_AVAILABLE = torch.cuda.is_available()
    logger.info(f"🎯 GPU Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
except ImportError:
    logger.warning("⚠️ GPU packages not installed. Install with: pip install torch sentence-transformers")
    GPU_AVAILABLE = False

# 🎯 EMBEDDING MODEL CONFIGURATION FOR GTX 1070
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Optimized Knowledge Acquisition Pipeline")
    parser.add_argument("--1024", dest="use_1024", action="store_true", 
                       help="Use 1024-dimensional embeddings (all-roberta-large-v1)")
    parser.add_argument("--model", type=str, default="phi4-mini-reasoning",
                       choices=["llama3.2:1b", "gemma3", "phi4-mini-reasoning", "granite3-dense"],
                       help="LLM model for reasoning and query generation")
    parser.add_argument("--topic", type=str, 
                       default="High Quality datasets for different types of AI / LLM tasks",
                       help="Topic to research")
    parser.add_argument("--duration", type=int, default=15,
                       help="Duration in minutes")
    parser.add_argument("--collection", type=str, default="datasets_knowledge",
                       help="Base collection name")
    parser.add_argument("--openai", action="store_true", help="Use OpenAI GPT-4o for reasoning (requires OPENAI_API_KEY)")
    return parser.parse_args()

# Parse arguments first
ARGS = parse_arguments()

EMBEDDING_MODELS = {
    768: {
        "name": "all-mpnet-base-v2",
        "dimensions": 768,
        "size_mb": 420,
        "description": "Best balance of quality and speed for 768D",
        "batch_size": 32,
        "collection_suffix": ""
    },
    1024: {
        "name": "all-roberta-large-v1", 
        "dimensions": 1024,
        "size_mb": 1400,
        "description": "Highest quality embeddings for maximum nuance",
        "batch_size": 16,  # Smaller batch for larger model
        "collection_suffix": "_1024"
    }
}

# Select model configuration based on arguments
EMBEDDING_CONFIG = EMBEDDING_MODELS[1024 if ARGS.use_1024 else 768]

# Model quality descriptions for GTX 1070
LLM_MODELS = {
    "llama3.2:1b": {"vram_gb": 2, "quality": "⭐⭐", "speed": "⭐⭐⭐⭐⭐", "knowledge_tasks": "⭐⭐"},
    "granite3-dense": {"vram_gb": 2.5, "quality": "⭐⭐⭐", "speed": "⭐⭐⭐⭐", "knowledge_tasks": "⭐⭐⭐"},
    "gemma3": {"vram_gb": 4.5, "quality": "⭐⭐⭐⭐", "speed": "⭐⭐⭐", "knowledge_tasks": "⭐⭐⭐⭐"},
    "phi4-mini-reasoning": {"vram_gb": 4.5, "quality": "⭐⭐⭐⭐⭐", "speed": "⭐⭐⭐", "knowledge_tasks": "⭐⭐⭐⭐⭐"}
}

logger.info(f"🎯 Configuration:")
logger.info(f"   Topic: {ARGS.topic}")
logger.info(f"   LLM Model: {ARGS.model}")
logger.info(f"   LLM Quality: {LLM_MODELS[ARGS.model]['quality']} | Knowledge Tasks: {LLM_MODELS[ARGS.model]['knowledge_tasks']}")
logger.info(f"   LLM VRAM: ~{LLM_MODELS[ARGS.model]['vram_gb']}GB")
logger.info(f"   Embedding Model: {EMBEDDING_CONFIG['name']}")
logger.info(f"   Embedding Dimensions: {EMBEDDING_CONFIG['dimensions']}")
logger.info(f"   Embedding VRAM: ~{EMBEDDING_CONFIG['size_mb']}MB")
logger.info(f"   Total VRAM Estimate: ~{LLM_MODELS[ARGS.model]['vram_gb'] + EMBEDDING_CONFIG['size_mb']/1000:.1f}GB")
logger.info(f"   Collection: {ARGS.collection}{EMBEDDING_CONFIG['collection_suffix']}")

class OptimizedKnowledgeAcquisitionPipeline:
    """GPU-optimized pipeline for GTX 1070"""
    
    def __init__(self, topic: str, collection_name: str, use_multi_provider: bool = True):
        self.topic = topic
        self.ollama = AsyncClient()
        
        # Use global configurations
        self.embedding_config = EMBEDDING_CONFIG
        self.embedding_dimension = self.embedding_config['dimensions']
        self.llm_model = ARGS.model  # Use selected LLM model
        
        logger.info(f"🧠 Using LLM: {self.llm_model}")
        logger.info(f"   Quality rating: {LLM_MODELS[self.llm_model]['knowledge_tasks']} for knowledge tasks")
        
        # Initialize GPU embedder if available
        self.gpu_embedder = None
        if GPU_AVAILABLE:
            self._init_gpu_embedder()
        
        # Optimized ChromaDB settings
        self.chroma_client = chromadb.PersistentClient(path="./vector_db")
        
        # Check if collection exists and has different embedding dimension
        collection_created = False
        try:
            existing_collection = self.chroma_client.get_collection(name=collection_name)
            # If collection exists, check if it's compatible
            existing_data = existing_collection.peek(limit=1)
            if existing_data['embeddings'] and len(existing_data['embeddings']) > 0:
                existing_dim = len(existing_data['embeddings'][0])
                if existing_dim != self.embedding_dimension:
                    logger.warning(f"Existing collection has {existing_dim}D embeddings, expected {self.embedding_dimension}D")
                    logger.info(f"🔄 Creating new collection: {collection_name}_optimized")
                    collection_name = f"{collection_name}_optimized"
                    self.collection = self.chroma_client.get_or_create_collection(
                        name=collection_name, 
                        metadata={"topic": topic, "optimized": True, "embedding_dim": self.embedding_dimension}
                    )
                    collection_created = True
                else:
                    self.collection = existing_collection
                    collection_created = True
            else:
                self.collection = existing_collection
                collection_created = True
        except Exception:
            # Collection doesn't exist, will be created below
            pass
        
        if not collection_created:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name, 
                metadata={"topic": topic, "optimized": True, "embedding_dim": self.embedding_dimension}
            )
        
        self.processed_urls = set()
        self.knowledge_coverage = defaultdict(int)
        self.search_history = []
        self.search_manager = None
        
        # Performance optimization settings
        self.embedding_batch = []
        self.batch_size = self.embedding_config['batch_size'] if GPU_AVAILABLE else 10
        self.max_concurrent_extractions = 8
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.performance_stats = {
            'embeddings_generated': 0,
            'content_extracted': 0,
            'total_processing_time': 0.0,
            'gpu_batches_processed': 0
        }
        
        # OpenAI settings
        self.use_openai = getattr(ARGS, 'openai', False)
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        # In-memory cache for per-source chunk counts
        self.source_chunk_counts = defaultdict(int)
        
        if use_multi_provider:
            self._init_multi_provider_search()
    
    def _init_gpu_embedder(self):
        """Initialize GPU-accelerated embedding model based on configuration"""
        try:
            model_name = self.embedding_config['name']
            self.gpu_embedder = SentenceTransformer(model_name, device='cuda')
            self.gpu_embedder.max_seq_length = 512  # Optimize for GTX 1070
            
            # Set memory optimization
            torch.cuda.empty_cache()
            logger.info(f"✅ GPU embedder loaded: {model_name}")
            logger.info(f"   Dimensions: {self.embedding_config['dimensions']}")
            logger.info(f"   Description: {self.embedding_config['description']}")
            
        except Exception as e:
            logger.error(f"❌ GPU embedder failed: {e}")
            self.gpu_embedder = None
    
    def _init_multi_provider_search(self):
        """Initialize search providers (now includes Tavily and Serper if API keys are set)"""
        providers = []
        try:
            ddg_provider = DuckDuckGoProvider()
            providers.append(ddg_provider)
            logger.info("✓ Added DuckDuckGo provider")
        except Exception as e:
            logger.warning(f"✗ Could not add DuckDuckGo: {e}")
        # Brave
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if brave_key:
            try:
                brave_provider = BraveSearchProvider(brave_key)
                providers.append(brave_provider)
                logger.info("✓ Added Brave Search provider")
            except Exception as e:
                logger.warning(f"✗ Could not add Brave Search: {e}")
        # Tavily
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            try:
                from multi import TavilyProvider
                tavily_provider = TavilyProvider(tavily_key)
                providers.append(tavily_provider)
                logger.info("✓ Added Tavily provider")
            except Exception as e:
                logger.warning(f"✗ Could not add Tavily: {e}")
        # Serper
        serper_key = os.environ.get("SERPER_API_KEY")
        if serper_key:
            try:
                from multi import SerperAPIProvider
                serper_provider = SerperAPIProvider(serper_key)
                providers.append(serper_provider)
                logger.info("✓ Added Serper.dev provider")
            except Exception as e:
                logger.warning(f"✗ Could not add Serper.dev: {e}")
        if providers:
            self.search_manager = SearchProviderManager(
                providers=providers, 
                strategy=RotationStrategy.WEIGHTED
            )
            logger.info(f"🔍 Initialized {len(providers)} search providers")
    
    async def generate_focused_queries(self, max_queries: int = 3) -> List[str]:
        """Generate focused, high-quality queries"""
        
        # Quality-focused query templates
        templates = [
            f"{self.topic} comprehensive guide",
            f"{self.topic} research dataset",
            f"{self.topic} academic paper",
            f"{self.topic} github repository",
            f"{self.topic} technical documentation",
            f"{self.topic} industry benchmark",
            f"{self.topic} latest 2025"
        ]
        
        # Rotate through templates to avoid repetition
        start_idx = len(self.search_history) % len(templates)
        selected_queries = templates[start_idx:start_idx + max_queries]
        
        # Fill remaining slots if needed
        if len(selected_queries) < max_queries:
            remaining = max_queries - len(selected_queries)
            selected_queries.extend(templates[:remaining])
        
        self.search_history.extend(selected_queries)
        return selected_queries
    
    def filter_quality_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter for high-quality sources relevant to AI/ML datasets"""
        
        # Premium domains for AI/ML content
        quality_domains = [
            'arxiv.org', 'paperswithcode.com', 'huggingface.co', 'github.com',
            'kaggle.com', 'scholar.google.com', 'ieee.org', 'acm.org',
            'openai.com', 'deepmind.com', 'ai.google', 'research.google',
            'microsoft.com', 'nvidia.com', 'tensorflow.org', 'pytorch.org'
        ]
        
        filtered_results = []
        domain_counts = {}
        
        for result in results:
            try:
                url = result.get('url', '')
                if not url:
                    continue
                
                # Extract domain
                domain = url.split('/')[2] if len(url.split('/')) > 2 else ''
                
                # Check if it's a quality domain
                is_quality = any(qd in domain.lower() for qd in quality_domains)
                
                # Limit results per domain for diversity
                current_count = domain_counts.get(domain, 0)
                max_per_domain = 3 if is_quality else 1
                
                if current_count < max_per_domain:
                    filtered_results.append(result)
                    domain_counts[domain] = current_count + 1
                
                # Limit total results for performance
                if len(filtered_results) >= 10:
                    break
                    
            except Exception as e:
                logger.warning(f"Error filtering result: {e}")
                continue
        
        logger.info(f"📊 Filtered to {len(filtered_results)} quality sources from {len(results)} results")
        return filtered_results
    
    async def parallel_content_extraction(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Extract content from multiple URLs in parallel"""
        # Filter already processed URLs and ensure all are strings
        new_urls = [url for url in urls if isinstance(url, str) and url and url not in self.processed_urls]
        if not new_urls:
            return []
        logger.info(f"📥 Extracting content from {len(new_urls)} URLs in parallel...")
        semaphore = asyncio.Semaphore(self.max_concurrent_extractions)
        async def extract_with_semaphore(url):
            async with semaphore:
                return await self.extract_content_optimized(url)
        tasks = [extract_with_semaphore(url) for url in new_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_extractions = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"❌ Failed {new_urls[i]}: {str(result)[:50]}")
                continue
            if result and isinstance(result, dict) and len(result.get('content', '')) > 200:
                successful_extractions.append(result)
        self.performance_stats['content_extracted'] += len(successful_extractions)
        logger.info(f"✅ Successfully extracted {len(successful_extractions)} contents")
        return successful_extractions

    async def extract_content_optimized(self, url: str) -> Optional[Dict[str, Any]]:
        """Optimized content extraction with PDF and encoding handling"""
        
        if url in self.processed_urls:
            return None
        
        # Skip low-quality domains
        skip_domains = ['baidu.com', 'weibo.com', 'facebook.com', 'twitter.com', 'tiktok.com']
        if any(domain in url.lower() for domain in skip_domains):
            self.processed_urls.add(url)
            return None
        
        try:
            if url.lower().endswith('.pdf'):
                # PDF extraction
                pdf_content = await self._extract_pdf_content(url)
                if pdf_content:
                    self.processed_urls.add(url)
                    return {'url': url, 'title': url.split('/')[-1], 'content': pdf_content}
                else:
                    self.processed_urls.add(url)
                    return None
            
            timeout = aiohttp.ClientTimeout(total=6)  # Quick timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        self.processed_urls.add(url)
                        return None
                    
                    # Try to detect encoding
                    raw = await response.read()
                    detected = chardet.detect(raw)
                    encoding = detected['encoding'] or 'utf-8'
                    try:
                        html = raw.decode(encoding, errors='replace')
                    except Exception:
                        html = raw.decode('utf-8', errors='replace')
                    
                    # Use thread pool for CPU-intensive parsing
                    loop = asyncio.get_event_loop()
                    parsed_content = await loop.run_in_executor(
                        self.thread_pool,
                        self._parse_html_optimized,
                        html, url
                    )
                    
                    self.processed_urls.add(url)
                    return parsed_content
        
        except Exception as e:
            logger.error(f"❌ Error extracting {url}: {str(e)[:80]}")
            self.processed_urls.add(url)
            return None

    async def _extract_pdf_content(self, url: str) -> Optional[str]:
        """Download and extract text from a PDF URL using PyMuPDF"""
        try:
            timeout = aiohttp.ClientTimeout(total=12)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        return None
                    pdf_bytes = await response.read()
            with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            return text if len(text) > 200 else None
        except Exception as e:
            logger.error(f"❌ PDF extraction error for {url}: {str(e)[:80]}")
            return None

    def _parse_html_optimized(self, html: str, url: str) -> Optional[Dict[str, Any]]:
        """Optimized HTML parsing in thread pool"""
        
        try:
            soup = BeautifulSoup(html, "html.parser")
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else ""
            
            # Try to find main content
            main_content = None
            content_selectors = ['main', 'article', '.content', '.post-content']
            
            for selector in content_selectors:
                elem = soup.select_one(selector)
                if elem:
                    main_content = elem
                    break
            
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract clean text
            content = main_content.get_text(separator=' ', strip=True)
            
            # Quality filters
            if len(content) < 300:
                return None
            
            # Basic English detection
            ascii_ratio = sum(1 for c in content[:500] if ord(c) < 128) / min(500, len(content))
            if ascii_ratio < 0.8:
                return None
            
            # Limit content size for performance
            if len(content) > 15000:
                content = content[:15000]
            
            return {
                'url': url,
                'title': title,
                'content': content,
                'length': len(content)
            }
            
        except Exception as e:
            logger.error(f"❌ HTML parsing error: {e}")
            return None
    
    def smart_chunking(self, content: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Smart content chunking optimized for embeddings"""
        
        if len(content) <= chunk_size:
            return [content]
        
        # Split on sentences for better semantic boundaries
        sentences = content.replace('. ', '.\n').split('\n')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + ' ' + sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + ' ' + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [content[:chunk_size]]
    
    async def batch_embed_and_store(self, content_items: List[Dict[str, Any]]):
        """Batch embedding generation and storage - key optimization"""
        all_chunks = []
        all_metadata = []
        all_embeddings = []
        # Prepare all chunks and metadata
        for item in content_items:
            if not item or not isinstance(item, dict):
                continue
            content = item.get('content', '')
            if not content:
                continue
            chunks = self.smart_chunking(content)
            for i, chunk in enumerate(chunks):
                metadata = {
                    'url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'timestamp': datetime.now().isoformat(),
                    'processing_mode': 'gpu_optimized' if self.gpu_embedder else 'cpu'
                }
                all_chunks.append(chunk)
                all_metadata.append(metadata)
        if not all_chunks:
            return
        logger.info(f"🔄 Processing {len(all_chunks)} chunks in batches of {self.batch_size}")
        for i in range(0, len(all_chunks), self.batch_size):
            start_time = time.time()
            batch_chunks = all_chunks[i:i + self.batch_size]
            batch_metadata = all_metadata[i:i + self.batch_size]
            # Generate embeddings for the batch
            if self.gpu_embedder:
                with torch.no_grad():
                    embeddings = self.gpu_embedder.encode(
                        batch_chunks, 
                        batch_size=self.batch_size,
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
                    embeddings_list = embeddings.cpu().numpy().tolist()
                torch.cuda.empty_cache()
            else:
                embedding_tasks = [
                    self.ollama.embeddings(model="nomic-embed-text", prompt=chunk)
                    for chunk in batch_chunks
                ]
                embedding_responses = await asyncio.gather(*embedding_tasks)
                embeddings_list = [resp["embedding"] for resp in embedding_responses if resp and "embedding" in resp]
            # Enhancement: filter and store only non-redundant, non-capped, non-outlier chunks
            for chunk, metadata, embedding in zip(batch_chunks, batch_metadata, embeddings_list):
                # Redundancy check
                if self.is_redundant_chunk(embedding):
                    logger.info(f"🔇 Skipping redundant chunk: {chunk[:60]}...")
                    continue
                # Source cap check
                url = metadata.get('url', '')
                if self.should_cap_source(url):
                    logger.info(f"🚫 Source cap reached, skipping chunk from: {url}")
                    continue
                # Outlier check
                if self.chunk_outlier(chunk):
                    logger.warning(f"⚠️ Chunk size outlier detected, skipping chunk: {chunk[:60]}...")
                    continue
                # Store the chunk
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata],
                    ids=[hashlib.md5(chunk.encode()).hexdigest()]
                )
                # Update in-memory source chunk count
                self.source_chunk_counts[url] += 1
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats['embeddings_generated'] += len(batch_chunks)
            self.performance_stats['gpu_batches_processed'] += 1
            self.performance_stats['total_processing_time'] += float(processing_time)
            throughput = len(batch_chunks) / processing_time if processing_time > 0 else 0
            logger.info(f"💾 Stored batch: {len(batch_chunks)} embeddings in {processing_time:.2f}s ({throughput:.1f} emb/sec)")
        # After storing, update cluster names/tags
        logger.info("🔎 Updating cluster names and tags...")
        cluster_info = await self.get_cluster_tags_and_names(n_clusters=8)
        if cluster_info:
            logger.info("🧩 Cluster themes and tags:")
            for cid, info in cluster_info.items():
                logger.info(f"  Cluster {cid+1}: {info['name']} | Tags: {', '.join(info['tags'])}")
    
    async def optimized_acquisition_loop(self, duration_minutes: int = 30):
        """Main optimized acquisition loop with fallback adaptive query refinement (no holistic)."""
        logger.info(f"🚀 Starting optimized acquisition for {duration_minutes} minutes")
        logger.info(f"   GPU Available: {GPU_AVAILABLE}")
        logger.info(f"   Batch Size: {self.batch_size}")
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        iteration = 0
        queries = await self.analyze_knowledge_gaps()
        while time.time() < end_time:
            iteration += 1
            iteration_start = time.time()
            logger.info(f"\n--- Optimized Iteration {iteration} ---")
            all_content_items = []
            all_search_results = []
            for query in queries:
                logger.info(f"🔍 Searching: {query}")
                if self.search_manager:
                    results = await self.search_manager.search(
                        query,
                        max_results=12,
                        use_multiple=False
                    )
                    search_results = []
                    for r in results:
                        if isinstance(r, dict):
                            search_results.append(r)
                        elif hasattr(r, 'to_dict'):
                            search_results.append(r.to_dict())
                else:
                    search_results = []
                if not search_results:
                    continue
                all_search_results.extend(search_results)
                quality_results = self.filter_quality_sources(search_results)
                # Ensure all URLs are strings and not None
                urls = [r.get('url') for r in quality_results if isinstance(r.get('url'), str) and r.get('url') is not None]
                content_items = await self.parallel_content_extraction(urls)
                all_content_items.extend(content_items)
            if all_content_items:
                await self.batch_embed_and_store(all_content_items)
            iteration_time = time.time() - iteration_start
            logger.info(f"⏱️  Iteration {iteration} completed in {iteration_time:.1f}s")
            queries = await self.analyze_knowledge_gaps()
            await asyncio.sleep(10)
        total_time = time.time() - start_time
        self.print_performance_summary(total_time)

    async def run_pipeline(self):
        """Unified entry point for the main acquisition loop."""
        await self.optimized_acquisition_loop(duration_minutes=ARGS.duration)

    def print_performance_summary(self, total_time: float):
        """Print comprehensive performance summary"""
        stats = self.performance_stats
        logger.info(f"\n🎯 Performance Summary ({total_time:.1f}s total)")
        logger.info(f"   Embeddings Generated: {stats['embeddings_generated']}")
        logger.info(f"   Content Items Extracted: {stats['content_extracted']}")
        logger.info(f"   GPU Batches Processed: {stats['gpu_batches_processed']}")
        logger.info(f"   Total Processing Time: {stats['total_processing_time']:.1f}s")
        if stats['embeddings_generated'] > 0:
            avg_emb_time = stats['total_processing_time'] / stats['gpu_batches_processed'] if stats['gpu_batches_processed'] > 0 else 0
            throughput = stats['embeddings_generated'] / total_time
            logger.info(f"   Average Embedding Throughput: {throughput:.1f} embeddings/second")
            logger.info(f"   Average Batch Time: {avg_emb_time:.2f}s")
        collection_data = self.collection.get()
        logger.info(f"   Total Stored Chunks: {len(collection_data['ids'])}")
        logger.info(f"   Unique Sources: {len(self.processed_urls)}")
        if GPU_AVAILABLE:
            logger.info(f"   GPU Memory Used: {torch.cuda.max_memory_allocated() / 1024**3:.2f}GB")

    async def llm_chat(self, prompt: str, temperature: float = 0.3, timeout: float = 60.0, retries: int = 2) -> str:
        logger.info("[DEBUG] Entering llm_chat (original client restore)...")
        logger.info(f"[DEBUG] Prompt length: {len(prompt)} | Prompt preview: {prompt[:120]}")
        last_error = None
        for attempt in range(1, retries + 2):
            try:
                if getattr(self, 'use_openai', False) and getattr(self, 'openai_api_key', None):
                    import openai
                    openai.api_key = self.openai_api_key
                    loop = asyncio.get_event_loop()
                    logger.info(f"[llm_chat] [OpenAI] Sending prompt (attempt {attempt})... (timeout={timeout}s)")
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: openai.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=temperature
                            )
                        ),
                        timeout=timeout
                    )
                    logger.info("[llm_chat] [OpenAI] Response received.")
                    return response.choices[0].message.content
                else:
                    logger.info(f"🟩 Using local LLM: {self.llm_model} for reasoning (attempt {attempt}) [original client]")
                    response = await asyncio.wait_for(
                        self.ollama.chat(
                            model=self.llm_model,
                            messages=[{"role": "user", "content": prompt}],
                            options={"temperature": temperature}
                        ),
                        timeout=timeout
                    )
                    logger.info(f"[llm_chat] [Ollama client] Response received: {str(response)[:120]}")
                    return response["message"]["content"]
            except Exception as e:
                logger.error(f"❌ LLM chat error (attempt {attempt}): {e}")
                last_error = str(e)
            await asyncio.sleep(5 * attempt)
            if attempt == 1 and timeout < 180:
                timeout = 180.0
        logger.error(f"[DEBUG] LLM chat failed after {retries+1} attempts. Last error: {last_error}")
        return f"Error: LLM chat failed after {retries+1} attempts ({last_error})"

    # NOTE: The linter may not recognize adaptive_query_refinement, but it is defined below and correct as of 2025.
    async def analyze_knowledge_gaps(self) -> List[str]:
        """Adaptive, LLM-driven query refinement for knowledge gap analysis. Falls back to static prompt if LLM fails."""
        logger.info("[DEBUG] Entering analyze_knowledge_gaps (ADAPTIVE QUERY REFINEMENT)...")
        try:
            # Use the topic as the base query
            base_query = self.topic
            # Optionally, get recent search results for context
            recent_results = []
            if self.search_manager and self.search_history:
                # Try to get results for the last query
                last_query = self.search_history[-1]
                results = await self.search_manager.search(
                    last_query,
                    max_results=8,
                    use_multiple=False
                )
                # Only include dicts or objects with to_dict()
                recent_results = []
                for r in results:
                    if isinstance(r, dict):
                        recent_results.append(r)
                    elif hasattr(r, 'to_dict'):
                        recent_results.append(r.to_dict())
            # Use adaptive query refinement
            queries = await self.adaptive_query_refinement(base_query, recent_results)
            if queries and all(isinstance(q, str) and len(q) > 10 for q in queries):
                logger.info(f"[DEBUG] Adaptive gap analysis returned {len(queries)} queries.")
                return queries[:3]
            else:
                logger.info("[DEBUG] Adaptive LLM returned no queries, using static fallback.")
                return await self.generate_focused_queries(max_queries=3)
        except Exception as e:
            logger.error(f"❌ Adaptive gap analysis error: {e}")
            return await self.generate_focused_queries(max_queries=3)

    async def query_knowledge_base(self, question: str, n_results: int = 5) -> str:
        """Query the optimized knowledge base"""
        results = self.collection.query(query_texts=[question], n_results=n_results)
        docs = results.get('documents') if results else None
        if not docs or not isinstance(docs, list) or not docs[0]:
            return "No relevant information found in the knowledge base."
        documents = docs[0] if isinstance(docs[0], list) else []
        # Defensive: handle NoneType for metadatas
        metadatas = []
        if results and results.get('metadatas') and isinstance(results['metadatas'], list) and results['metadatas']:
            try:
                first_meta = results['metadatas'][0]
                if isinstance(first_meta, list):
                    metadatas = first_meta
                else:
                    metadatas = []
            except Exception:
                metadatas = []
        context = "\n\n".join(documents)
        prompt = f"""Based on the following information, provide a comprehensive answer to: {question}\n\nContext:\n{context[:4000]}  # Limit context for speed\n\nProvide a clear, structured response."""
        return await self.llm_chat(prompt, temperature=0.3)
    
    async def adaptive_query_refinement(self, base_query: str, search_results: List[Dict]) -> List[str]:
        """Refine queries based on initial search results to get better coverage."""
        if not search_results:
            return [base_query]
        try:
            # Analyze what the search returned
            titles = [r.get('title', '') for r in search_results[:10]]
            snippets = [r.get('snippet', '') for r in search_results[:5]]
            prompt = f"""Analyze these search results for the query '{base_query}' and generate 3 refined follow-up queries.\n\nORIGINAL QUERY: {base_query}\n\nSEARCH RESULTS ANALYSIS:\nTitles found: {'; '.join(titles)}\nContent snippets: {'; '.join(snippets[:3])}\n\nBased on these results:\n1. What important aspects of the topic seem under-represented?\n2. What specific angles need deeper investigation? \n3. What authoritative sources might have been missed?\n\nGenerate 3 refined search queries that will capture missing information:\n1. [refined query 1]\n2. [refined query 2]\n3. [refined query 3]\n\nMake queries specific, actionable, and different from what was already found."""
            content = await self.llm_chat(prompt, temperature=0.3)
            refined_queries = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                    query = line.split('. ', 1)[-1].strip()
                    query = query.replace('[', '').replace(']', '')
                    if len(query) > 10:
                        refined_queries.append(query)
            return refined_queries[:3] if refined_queries else [base_query]
        except Exception as e:
            logger.error(f"❌ Query refinement error: {e}")
            return [base_query]

    def chunk_outlier(self, chunk: str, min_len: int = 100, max_len: int = 1500) -> bool:
        """Detect if a chunk is an outlier in size or content using length and simple heuristics."""
        l = len(chunk)
        if l < min_len or l > max_len:
            return True
        # Optionally, add more advanced outlier detection here (e.g., language detection, entropy, etc.)
        return False

    async def get_cluster_tags_and_names(self, n_clusters: int = 8) -> dict:
        """Cluster all embeddings, use LLM to name and tag each cluster, and return mapping."""
        try:
            data = self.collection.get()
            embeds = data.get("embeddings") or []
            docs = data.get("documents") or []
            import numpy as np
            embeds = np.array(embeds)
            if embeds.size == 0 or len(docs) == 0:
                return {}
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            clusters = kmeans.fit_predict(embeds)
            cluster_map = {i: [] for i in range(n_clusters)}
            for idx, c in enumerate(clusters):
                cluster_map[c].append(docs[idx])
            cluster_info = {}
            for c, chunk_list in cluster_map.items():
                sample = '\n'.join(chunk_list[:5])
                prompt = f"""Analyze these document excerpts and provide:\n1. A descriptive name for this cluster\n2. 3-5 relevant tags\n\nExcerpts:\n{sample}"""
                llm_resp = await self.llm_chat(prompt, temperature=0.3)
                lines = llm_resp.split('\n')
                name = lines[0].strip() if lines else f"Cluster {c+1}"
                tags = [l.strip('-• ') for l in lines[1:] if l.strip()]
                cluster_info[c] = {"name": name, "tags": tags}
            return cluster_info
        except Exception as e:
            logger.warning(f"[WARN] Cluster naming/tagging failed: {e}")
            return {}

    def is_redundant_chunk(self, new_embedding, threshold=0.95):
        try:
            results = self.collection.query(
                query_embeddings=[new_embedding],
                n_results=3,
                include=["distances"]
            )
            if results and results.get("distances"):
                distances = results.get("distances", [[]])[0]
                if distances and min(distances) < (1 - threshold):
                    return True
            return False
        except Exception as e:
            logger.warning(f"[WARN] Redundancy check (ANN) failed: {e}")
            return False

    def should_cap_source(self, url, cap=30):
        try:
            count = self.source_chunk_counts.get(url, 0)
            return count >= cap
        except Exception as e:
            logger.warning(f"[WARN] Source cap check (cache) failed: {e}")
            return False

# --- Script Entry Point ---
def main():
    """Main execution for standalone script."""
    logger.info("🚀 Starting Optimized Knowledge Acquisition Pipeline...")
    pipeline = OptimizedKnowledgeAcquisitionPipeline(
        topic=ARGS.topic,
        collection_name=ARGS.collection + EMBEDDING_CONFIG['collection_suffix'],
        use_multi_provider=True
    )
    # You may want to expose a method like pipeline.run() or similar for the main acquisition loop
    # For now, let's assume the main loop is a method called run_pipeline (implement if missing)
    if hasattr(pipeline, 'run_pipeline'):
        asyncio.run(pipeline.run_pipeline())
    else:
        logger.error("No run_pipeline() method found on pipeline. Please implement the main acquisition loop.")

if __name__ == "__main__":
    main()