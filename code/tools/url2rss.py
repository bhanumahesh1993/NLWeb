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

class URLToRSS:
    def __init__(self, base_url: str, max_depth: int = 2, max_pages: int = 100):
        """
        Initialize the URL to RSS converter.
        
        Args:
            base_url: The base URL to crawl
            max_depth: Maximum depth of pages to crawl
            max_pages: Maximum number of pages to process
        """
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls: Set[str] = set()
        self.pages: List[Dict[str, Any]] = []
        self.session = None
        
    async def init_session(self):
        """Initialize the aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
    async def close_session(self):
        """Close the aiohttp session."""
        if self.session:
            await self.session.close()
            self.session = None
            
    def is_valid_url(self, url: str) -> bool:
        """
        Check if a URL is valid and belongs to the same domain.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL is valid and belongs to the same domain
        """
        try:
            parsed_base = urlparse(self.base_url)
            parsed_url = urlparse(url)
            return (
                parsed_url.netloc == parsed_base.netloc and
                parsed_url.scheme in ['http', 'https'] and
                not url.endswith(('.pdf', '.jpg', '.png', '.gif', '.zip', '.exe'))
            )
        except Exception:
            return False
            
    async def fetch_page(self, url: str) -> str:
        """
        Fetch a page's content.
        
        Args:
            url: URL to fetch
            
        Returns:
            Page content as string
        """
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return ""
                return await response.text()
        except Exception:
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
        soup = BeautifulSoup(content, 'html.parser')
        links = []
        
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(base_url, href)
            if self.is_valid_url(absolute_url):
                links.append(absolute_url)
                
        return links
        
    def extract_page_info(self, content: str, url: str) -> Dict[str, Any]:
        """
        Extract page information.
        
        Args:
            content: Page content
            url: Page URL
            
        Returns:
            Dictionary containing page information
        """
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else url
        
        # Extract description
        description = ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            description = meta_desc['content']
        else:
            # Use first paragraph as description
            first_p = soup.find('p')
            if first_p:
                description = first_p.get_text()[:200] + "..."
                
        # Extract content
        content_text = ""
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
        if main_content:
            content_text = main_content.get_text()
        else:
            content_text = soup.get_text()
            
        return {
            'title': title,
            'description': description,
            'content': content_text,
            'url': url,
            'published': datetime.now().isoformat()
        }
        
    async def crawl(self, url: str, depth: int = 0):
        """
        Crawl a URL and its subpages.
        
        Args:
            url: URL to crawl
            depth: Current depth level
        """
        if depth > self.max_depth or len(self.pages) >= self.max_pages or url in self.visited_urls:
            return
            
        self.visited_urls.add(url)
        content = await self.fetch_page(url)
        
        if not content:
            return
            
        # Extract and store page information
        page_info = self.extract_page_info(content, url)
        self.pages.append(page_info)
        
        # Extract and crawl links
        if depth < self.max_depth:
            links = self.extract_links(content, url)
            for link in links:
                if link not in self.visited_urls:
                    await self.crawl(link, depth + 1)
                    
    def generate_rss(self, output_file: str):
        """
        Generate RSS feed from crawled pages.
        
        Args:
            output_file: Path to output RSS file
        """
        rss = ET.Element('rss', version='2.0')
        channel = ET.SubElement(rss, 'channel')
        
        # Add channel information
        ET.SubElement(channel, 'title').text = f"RSS Feed for {self.base_url}"
        ET.SubElement(channel, 'link').text = self.base_url
        ET.SubElement(channel, 'description').text = f"Generated RSS feed for {self.base_url}"
        ET.SubElement(channel, 'lastBuildDate').text = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
        
        # Add items
        for page in self.pages:
            item = ET.SubElement(channel, 'item')
            ET.SubElement(item, 'title').text = page['title']
            ET.SubElement(item, 'link').text = page['url']
            ET.SubElement(item, 'description').text = page['description']
            ET.SubElement(item, 'pubDate').text = page['published']
            ET.SubElement(item, 'guid').text = page['url']
            
        # Write to file
        xml_str = minidom.parseString(ET.tostring(rss)).toprettyxml(indent="  ")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(xml_str)
            
async def main():
    parser = argparse.ArgumentParser(description='Convert a URL to RSS feed')
    parser.add_argument('url', help='Base URL to crawl')
    parser.add_argument('--output', '-o', default='output.rss', help='Output RSS file path')
    parser.add_argument('--depth', '-d', type=int, default=2, help='Maximum crawl depth')
    parser.add_argument('--max-pages', '-m', type=int, default=100, help='Maximum number of pages to process')
    
    args = parser.parse_args()
    
    converter = URLToRSS(args.url, args.depth, args.max_pages)
    await converter.init_session()
    
    try:
        print(f"Crawling {args.url}...")
        await converter.crawl(args.url)
        print(f"Generating RSS feed...")
        converter.generate_rss(args.output)
        print(f"RSS feed generated: {args.output}")
    finally:
        await converter.close_session()
        
if __name__ == '__main__':
    asyncio.run(main()) 