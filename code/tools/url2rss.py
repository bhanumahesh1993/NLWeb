#!/usr/bin/env python3
# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Convert a generic URL into RSS feeds for its pages and subpages.
This script crawls a website and generates RSS feeds that can be used with db_load.
"""

import os
import sys
import json
import asyncio
import aiohttp
import argparse
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Set
import xml.etree.ElementTree as ET
from xml.dom import minidom
import time
import re

class URLToRSS:
    def __init__(self, base_url: str, max_depth: int = 2, max_pages: int = 100, delay: float = 1.0):
        """
        Initialize the URL to RSS converter.
        
        Args:
            base_url: The base URL to crawl
            max_depth: Maximum depth of pages to crawl
            max_pages: Maximum number of pages to process
            delay: Delay between requests in seconds
        """
        # Ensure URL has a scheme
        if not base_url.startswith(('http://', 'https://')):
            base_url = 'https://' + base_url
            
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.pages: List[Dict[str, Any]] = []
        self.session = None
        self.domain = urlparse(base_url).netloc
        self.website_name = self._extract_website_name(base_url)
        self.sitemap_urls: List[str] = []
        
    def _extract_website_name(self, url: str) -> str:
        """
        Extract website name without TLD from URL.
        
        Args:
            url: Full URL
            
        Returns:
            Website name without TLD (e.g., 'tcpwave' from 'tcpwave.in')
        """
        domain = urlparse(url).netloc
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Split by dots and take the first part (before TLD)
        parts = domain.split('.')
        if len(parts) >= 2:
            return parts[0]
        return domain
        
    def normalize_url(self, url: str) -> str:
        """
        Normalize a URL by removing fragments and trailing slashes.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL
        """
        parsed = urlparse(url)
        # Remove fragment and rebuild URL
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        # Remove trailing slash unless it's the root
        if normalized.endswith('/') and len(parsed.path) > 1:
            normalized = normalized[:-1]
        return normalized
        
    async def init_session(self):
        """Initialize the aiohttp session with proper headers."""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
            
    async def close_session(self):
        """Close the aiohttp session."""
        if self.session:
            try:
                await self.session.close()
                # Give some time for the connection to close properly
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Warning: Error closing session: {e}")
            finally:
                self.session = None
    
    async def discover_sitemap_urls(self) -> List[str]:
        """
        Discover URLs from sitemap.xml files.
        
        Returns:
            List of URLs found in sitemaps (filtered and ready for crawling)
        """
        sitemap_urls = []
        all_sitemap_urls = []  # Keep track of ALL URLs found for saving to file
        
        # Common sitemap locations
        sitemap_locations = [
            f"{self.base_url}/sitemap.xml",
            f"{self.base_url}/sitemap_index.xml",
            f"{self.base_url}/sitemap/sitemap.xml",
            f"{self.base_url}/sitemaps/sitemap.xml",
            f"{self.base_url}/sitemaps.xml"
        ]
        
        print("🗺️  Checking for sitemaps...")
        
        # First check robots.txt for sitemap declarations
        robots_sitemaps = await self._check_robots_txt()
        if robots_sitemaps:
            sitemap_locations.extend(robots_sitemaps)
            print(f"Found {len(robots_sitemaps)} sitemap(s) in robots.txt")
        
        # Check each sitemap location
        for sitemap_url in sitemap_locations:
            urls_from_sitemap = await self._parse_sitemap(sitemap_url)
            if urls_from_sitemap:
                all_sitemap_urls.extend(urls_from_sitemap)  # Save ALL URLs found
                sitemap_urls.extend(urls_from_sitemap)
                print(f"Found {len(urls_from_sitemap)} URLs in {sitemap_url}")
        
        # Store all URLs for saving to file later
        self.all_sitemap_urls = list(set(all_sitemap_urls))  # Remove duplicates but keep all
        
        # Remove duplicates and validate URLs for crawling
        unique_urls = []
        seen = set()
        valid_count = 0
        invalid_count = 0
        
        for url in sitemap_urls:
            normalized_url = self.normalize_url(url)
            if normalized_url not in seen:
                seen.add(normalized_url)
                if self.is_valid_url(normalized_url):
                    unique_urls.append(normalized_url)
                    valid_count += 1
                else:
                    invalid_count += 1
        
        print(f"📍 Total URLs found in sitemaps: {len(self.all_sitemap_urls)}")
        print(f"📍 Valid URLs for crawling: {valid_count}")
        print(f"📍 Invalid/filtered URLs: {invalid_count}")
        
        return unique_urls[:self.max_pages]  # Limit to max_pages
    
    async def _check_robots_txt(self) -> List[str]:
        """
        Check robots.txt for sitemap declarations.
        
        Returns:
            List of sitemap URLs found in robots.txt
        """
        robots_url = f"{self.base_url}/robots.txt"
        sitemap_urls = []
        
        try:
            content = await self.fetch_page(robots_url, allow_non_html=True)
            if content:
                # Look for sitemap declarations
                for line in content.split('\n'):
                    line = line.strip()
                    if line.lower().startswith('sitemap:'):
                        sitemap_url = line.split(':', 1)[1].strip()
                        sitemap_urls.append(sitemap_url)
        except Exception as e:
            print(f"Could not fetch robots.txt: {e}")
        
        return sitemap_urls
    
    async def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """
        Parse a sitemap XML file and extract URLs.
        
        Args:
            sitemap_url: URL of the sitemap to parse
            
        Returns:
            List of URLs found in the sitemap
        """
        try:
            content = await self.fetch_page(sitemap_url, allow_non_html=True)
            if not content:
                return []
            
            # Parse XML
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                print(f"Invalid XML in sitemap: {sitemap_url}")
                return []
            
            urls = []
            
            # Handle sitemap index files (containing references to other sitemaps)
            if 'sitemapindex' in root.tag.lower():
                for sitemap in root:
                    loc_elem = sitemap.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None and loc_elem.text:
                        # Recursively parse nested sitemaps
                        nested_urls = await self._parse_sitemap(loc_elem.text)
                        urls.extend(nested_urls)
            
            # Handle regular sitemap files (containing URLs)
            else:
                for url_elem in root:
                    loc_elem = url_elem.find('.//{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
                    if loc_elem is not None and loc_elem.text:
                        urls.append(loc_elem.text)
            
            return urls
            
        except Exception as e:
            print(f"Error parsing sitemap {sitemap_url}: {e}")
            return []

    def save_sitemap_file(self, sitemap_urls: List[str] = None):
        """
        Save discovered sitemap URLs to a file.
        
        Args:
            sitemap_urls: List of URLs to save (optional, defaults to all_sitemap_urls)
        """
        # Use all_sitemap_urls if available, otherwise fall back to provided list
        urls_to_save = getattr(self, 'all_sitemap_urls', sitemap_urls or [])
        
        if not urls_to_save:
            print("No sitemap URLs to save")
            return
        
        filename = f"{self.website_name}_sitemap.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Sitemap URLs for {self.domain}\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total URLs: {len(urls_to_save)}\n")
                
                # Group URLs by domain for better organization
                grouped_urls = {}
                for url in urls_to_save:
                    domain = urlparse(url).netloc
                    if domain not in grouped_urls:
                        grouped_urls[domain] = []
                    grouped_urls[domain].append(url)
                
                f.write(f"# Domains found: {', '.join(sorted(grouped_urls.keys()))}\n\n")
                
                # Write URLs grouped by domain
                for domain in sorted(grouped_urls.keys()):
                    f.write(f"# URLs from {domain} ({len(grouped_urls[domain])} URLs)\n")
                    for url in sorted(grouped_urls[domain]):
                        f.write(f"{url}\n")
                    f.write("\n")
            
            print(f"💾 Sitemap URLs saved to: {filename}")
            print(f"💾 Total URLs saved: {len(urls_to_save)}")
            
        except Exception as e:
            print(f"Error saving sitemap file: {e}")
            
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and belongs to the same domain.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is valid and belongs to the same domain
        """
        try:
            parsed_url = urlparse(url)
            
            # Check if it has a valid scheme
            if parsed_url.scheme not in ['http', 'https']:
                return False
            
            # More flexible domain matching - handle www and subdomain variants
            url_domain = parsed_url.netloc.lower()
            base_domain = self.domain.lower()
            
            # Remove www. prefix for comparison
            if url_domain.startswith('www.'):
                url_domain = url_domain[4:]
            if base_domain.startswith('www.'):
                base_domain = base_domain[4:]
            
            # Check if domains match (exact match or subdomain of base domain)
            if not (url_domain == base_domain or url_domain.endswith('.' + base_domain)):
                return False
                
            # Skip file downloads
            if url.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.exe', '.doc', '.docx', '.mp4', '.avi', '.mov')):
                return False
                
            # Skip common non-content paths
            path_lower = parsed_url.path.lower()
            skip_paths = ['/admin', '/login', '/logout', '/api', '/wp-admin', '/wp-login']
            if any(skip_path in path_lower for skip_path in skip_paths):
                return False
                
            return True
        except Exception as e:
            print(f"Error validating URL {url}: {e}")
            return False
            
    async def fetch_page(self, url: str, allow_non_html: bool = False) -> str:
        """
        Fetch a page's content.
        
        Args:
            url: URL to fetch
            allow_non_html: Whether to allow non-HTML content types
            
        Returns:
            Page content as string
        """
        try:
            print(f"Fetching: {url}")
            async with self.session.get(url) as response:
                if response.status != 200:
                    print(f"HTTP {response.status} for {url}")
                    return ""
                    
                content_type = response.headers.get('content-type', '').lower()
                
                # Allow XML and text content for sitemaps and robots.txt
                if not allow_non_html and 'text/html' not in content_type:
                    print(f"Skipping non-HTML content: {content_type} for {url}")
                    return ""
                    
                content = await response.text()
                print(f"Successfully fetched {len(content)} characters from {url}")
                return content
                
        except asyncio.TimeoutError:
            print(f"Timeout fetching {url}")
            return ""
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""
            
    def extract_links(self, content: str, base_url: str) -> List[str]:
        """
        Extract links from page content.
        
        Args:
            content: Page content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of absolute URLs
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            links = set()  # Use set to avoid duplicates
            
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                
                # Skip empty hrefs, anchors, and javascript links
                if not href or href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                    continue
                    
                absolute_url = urljoin(base_url, href)
                normalized_url = self.normalize_url(absolute_url)
                
                if self.is_valid_url(normalized_url):
                    links.add(normalized_url)
                    
            return list(links)
        except Exception as e:
            print(f"Error extracting links from {base_url}: {e}")
            return []
        
    def extract_page_info(self, content: str, url: str) -> Dict[str, Any]:
        """
        Extract page information.
        
        Args:
            content: Page content
            url: Page URL
            
        Returns:
            Dictionary containing page information
        """
        try:
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract title
            title = "Untitled Page"
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.find('h1'):
                title = soup.find('h1').get_text().strip()
                
            # Extract description
            description = ""
            
            # Try meta description first
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc['content'].strip()
            else:
                # Try Open Graph description
                og_desc = soup.find('meta', attrs={'property': 'og:description'})
                if og_desc and og_desc.get('content'):
                    description = og_desc['content'].strip()
                else:
                    # Use first paragraph as description
                    paragraphs = soup.find_all('p')
                    for p in paragraphs:
                        text = p.get_text().strip()
                        if len(text) > 50:  # Only use substantial paragraphs
                            description = text[:300] + "..." if len(text) > 300 else text
                            break
                            
            # Extract main content
            content_text = ""
            
            # Try to find main content areas
            main_selectors = [
                'main', 
                'article', 
                '.content', 
                '.main-content', 
                '#content',
                '.post-content',
                '.entry-content'
            ]
            
            main_content = None
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break
                    
            if main_content:
                content_text = main_content.get_text(strip=True)
            else:
                # Remove navigation, footer, and sidebar content
                for tag in soup(['nav', 'footer', 'aside', 'header']):
                    tag.decompose()
                content_text = soup.get_text(strip=True)
                
            # Clean up content
            content_text = ' '.join(content_text.split())  # Normalize whitespace
            
            return {
                'title': title,
                'description': description,
                'content': content_text[:1000] + "..." if len(content_text) > 1000 else content_text,
                'url': url,
                'published': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error extracting page info from {url}: {e}")
            return {
                'title': f"Page from {url}",
                'description': f"Content from {url}",
                'content': "",
                'url': url,
                'published': datetime.now().isoformat()
            }
    
    async def crawl_from_sitemap(self, sitemap_urls: List[str]):
        """
        Crawl pages from sitemap URLs directly.
        
        Args:
            sitemap_urls: List of URLs from sitemap to crawl
        """
        print(f"🚀 Crawling {min(len(sitemap_urls), self.max_pages)} pages from sitemap...")
        
        # Limit to max_pages
        urls_to_crawl = sitemap_urls[:self.max_pages]
        
        for i, url in enumerate(urls_to_crawl, 1):
            if len(self.pages) >= self.max_pages:
                break
                
            normalized_url = self.normalize_url(url)
            if normalized_url in self.visited_urls:
                continue
                
            self.visited_urls.add(normalized_url)
            print(f"Crawling ({i}/{len(urls_to_crawl)}): {normalized_url}")
            
            content = await self.fetch_page(normalized_url)
            
            if content:
                page_info = self.extract_page_info(content, normalized_url)
                self.pages.append(page_info)
                print(f"✅ Added page: {page_info['title']}")
            else:
                print(f"❌ No content retrieved from {normalized_url}")
            
            # Add delay between requests
            if self.delay > 0:
                await asyncio.sleep(self.delay)
            
    async def crawl(self, url: str, depth: int = 0):
        """
        Crawl a URL and its subpages.
        
        Args:
            url: URL to crawl
            depth: Current depth level
        """
        normalized_url = self.normalize_url(url)
        
        if (depth > self.max_depth or 
            len(self.pages) >= self.max_pages or 
            normalized_url in self.visited_urls):
            return
            
        self.visited_urls.add(normalized_url)
        print(f"Crawling (depth {depth}): {normalized_url}")
        
        content = await self.fetch_page(normalized_url)
        
        if not content:
            print(f"No content retrieved from {normalized_url}")
            return
            
        # Extract and store page information
        page_info = self.extract_page_info(content, normalized_url)
        self.pages.append(page_info)
        print(f"Added page: {page_info['title']}")
        
        # Add delay between requests to be respectful
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        
        # Extract and crawl links if we haven't reached max depth
        if depth < self.max_depth and len(self.pages) < self.max_pages:
            links = self.extract_links(content, normalized_url)
            print(f"Found {len(links)} valid links on {normalized_url}")
            
            for link in links:
                if link not in self.visited_urls and len(self.pages) < self.max_pages:
                    await self.crawl(link, depth + 1)
                    
    def generate_rss(self, output_file: str):
        """
        Generate RSS feed from crawled pages.
        
        Args:
            output_file: Path to output RSS file
        """
        print(f"Generating RSS feed with {len(self.pages)} pages...")
        
        # Register namespace for content:encoded
        ET.register_namespace('content', 'http://purl.org/rss/1.0/modules/content/')
        
        rss = ET.Element('rss', {
            'version': '2.0',
            'xmlns:content': 'http://purl.org/rss/1.0/modules/content/'
        })
        channel = ET.SubElement(rss, 'channel')
        
        # Add channel information
        ET.SubElement(channel, 'title').text = f"RSS Feed for {self.domain}"
        ET.SubElement(channel, 'link').text = self.base_url
        ET.SubElement(channel, 'description').text = f"Generated RSS feed for {self.domain}"
        ET.SubElement(channel, 'lastBuildDate').text = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
        ET.SubElement(channel, 'generator').text = "NLWeb url2rss Tool"
        
        # Add items
        for page in self.pages:
            item = ET.SubElement(channel, 'item')
            ET.SubElement(item, 'title').text = self._clean_text(page['title'])
            ET.SubElement(item, 'link').text = page['url']
            ET.SubElement(item, 'description').text = self._clean_text(page['description'])
            ET.SubElement(item, 'pubDate').text = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
            ET.SubElement(item, 'guid').text = page['url']
            
            # Add content if available - use a simpler approach
            if page['content']:
                # Add content in description if it's not already there
                if page['content'] not in page['description']:
                    full_description = f"{page['description']}\n\n{page['content']}"[:500] + "..."
                    ET.SubElement(item, 'summary').text = self._clean_text(full_description)
                
        # Write to file with better error handling
        try:
            rough_string = ET.tostring(rss, 'unicode', xml_declaration=True)
            
            # Write directly without reparsing to avoid XML namespace issues
            with open(output_file, 'w', encoding='utf-8') as f:
                # Format the XML manually for better readability
                formatted_xml = self._format_xml(rough_string)
                f.write(formatted_xml)
                
            print(f"RSS feed saved to {output_file}")
        except Exception as e:
            print(f"Error generating RSS: {e}")
            # Fallback: write simple XML without formatting
            self._generate_simple_rss(output_file)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for XML by escaping special characters."""
        if not text:
            return ""
        # Remove or replace problematic characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&apos;')
        return text.strip()
    
    def _format_xml(self, xml_string: str) -> str:
        """Simple XML formatting without using minidom."""
        lines = []
        indent = 0
        for line in xml_string.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('</'):
                indent -= 2
            lines.append(' ' * indent + line)
            if line.startswith('<') and not line.startswith('</') and not line.endswith('/>'):
                if not any(line.startswith(f'<{tag}') for tag in ['title', 'link', 'description', 'pubDate', 'guid', 'generator', 'lastBuildDate', 'summary']):
                    indent += 2
        return '\n'.join(lines)
    
    def _generate_simple_rss(self, output_file: str):
        """Generate a simple RSS feed without complex XML processing."""
        print("Generating simple RSS feed as fallback...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<rss version="2.0">\n')
            f.write('  <channel>\n')
            f.write(f'    <title>RSS Feed for {self.domain}</title>\n')
            f.write(f'    <link>{self.base_url}</link>\n')
            f.write(f'    <description>Generated RSS feed for {self.domain}</description>\n')
            f.write(f'    <lastBuildDate>{datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")}</lastBuildDate>\n')
            f.write('    <generator>NLWeb url2rss Tool</generator>\n')
            
            for page in self.pages:
                f.write('    <item>\n')
                f.write(f'      <title>{self._clean_text(page["title"])}</title>\n')
                f.write(f'      <link>{page["url"]}</link>\n')
                f.write(f'      <description>{self._clean_text(page["description"])}</description>\n')
                f.write(f'      <pubDate>{datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")}</pubDate>\n')
                f.write(f'      <guid>{page["url"]}</guid>\n')
                f.write('    </item>\n')
                
            f.write('  </channel>\n')
            f.write('</rss>\n')
            
        print(f"Simple RSS feed saved to {output_file}")
        
    def print_summary(self):
        """Print a summary of the crawling results."""
        print(f"\n=== Crawling Summary ===")
        print(f"Base URL: {self.base_url}")
        print(f"Website name: {self.website_name}")
        print(f"Pages crawled: {len(self.pages)}")
        print(f"URLs visited: {len(self.visited_urls)}")
        print(f"Max depth: {self.max_depth}")
        print(f"Max pages: {self.max_pages}")
        print(f"Sitemap URLs found: {len(self.sitemap_urls)}")
        
        if self.pages:
            print(f"\nPages found:")
            for i, page in enumerate(self.pages[:10], 1):  # Show first 10 pages
                print(f"  {i}. {page['title']} - {page['url']}")
            if len(self.pages) > 10:
                print(f"  ... and {len(self.pages) - 10} more pages")
        else:
            print("\nNo pages were successfully crawled!")
            
async def main():
    parser = argparse.ArgumentParser(description='Convert a URL to RSS feed')
    parser.add_argument('url', help='Base URL to crawl (can include or omit http://)')
    parser.add_argument('--output', '-o', default='output.rss', help='Output RSS file path')
    parser.add_argument('--depth', '-d', type=int, default=2, help='Maximum crawl depth')
    parser.add_argument('--max-pages', '-m', type=int, default=100, help='Maximum number of pages to process')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    parser.add_argument('--sitemap-only', action='store_true', help='Only use sitemap URLs (skip link crawling)')
    
    args = parser.parse_args()
    
    print(f"Starting crawl of {args.url}")
    print(f"Max depth: {args.depth}, Max pages: {args.max_pages}")
    print(f"Delay between requests: {args.delay}s")
    print(f"Sitemap-only mode: {args.sitemap_only}")
    print("-" * 50)
    
    converter = URLToRSS(args.url, args.depth, args.max_pages, args.delay)
    await converter.init_session()
    
    try:
        start_time = time.time()
        
        # First, try to discover URLs from sitemaps
        sitemap_urls = await converter.discover_sitemap_urls()
        converter.sitemap_urls = sitemap_urls
        
        # Save sitemap URLs to file
        converter.save_sitemap_file(sitemap_urls)
        
        # Choose crawling strategy
        if sitemap_urls and (args.sitemap_only or len(sitemap_urls) >= args.max_pages):
            print(f"🗺️  Using sitemap-based crawling ({len(sitemap_urls)} URLs available)")
            await converter.crawl_from_sitemap(sitemap_urls)
        else:
            print(f"🔗 Using link-based crawling (depth {args.depth})")
            if sitemap_urls:
                print(f"   Will supplement with {len(sitemap_urls)} sitemap URLs")
                # First crawl from sitemap
                await converter.crawl_from_sitemap(sitemap_urls[:args.max_pages//2])
            
            # Then do traditional crawling for remaining pages
            if len(converter.pages) < args.max_pages:
                await converter.crawl(args.url)
        
        end_time = time.time()
        
        converter.print_summary()
        print(f"\nCrawling completed in {end_time - start_time:.2f} seconds")
        
        if converter.pages:
            converter.generate_rss(args.output)
            print(f"\n✅ Success! RSS feed with {len(converter.pages)} pages generated: {args.output}")
        else:
            print("\n❌ No pages were crawled. Check the URL and try again.")
            print("Common issues:")
            print("  - Website blocks automated requests")
            print("  - Network connectivity problems") 
            print("  - Invalid URL format")
            
    except KeyboardInterrupt:
        print("\n⚠️  Crawling interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during crawling: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await converter.close_session()
        
if __name__ == '__main__':
    asyncio.run(main())