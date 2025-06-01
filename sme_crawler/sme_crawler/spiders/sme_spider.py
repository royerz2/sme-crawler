import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from datetime import datetime
import re

class SMESpider(CrawlSpider):
    name = 'sme_spider'
    
    allowed_domains = [
        'ondernemeninlimburg.nl',
        'sittard-geleen.nieuws.nl',
        'mkblimburg.nl',
        'brightlands.com',
        'pomlimburg.be',
        'business.gov.nl'
    ]
    
    start_urls = [
        'https://www.ondernemeninlimburg.nl',
        'https://www.sittard-geleen.nieuws.nl',
        'https://www.mkblimburg.nl',
        'https://www.brightlands.com',
        'https://www.pomlimburg.be',
        'https://www.business.gov.nl'
    ]
    
    rules = (
        Rule(
            LinkExtractor(
                allow_domains=allowed_domains,
                deny=(
                    r'\.(jpg|jpeg|png|gif|pdf|zip|doc|docx|xls|xlsx)$',
                    r'/tag/',
                    r'/author/',
                    r'/category/',
                    r'/wp-admin/',
                    r'/wp-login/',
                    r'/feed/',
                    r'/rss/',
                    r'/atom/',
                    r'/sitemap/',
                    r'/search/',
                    r'/login/',
                    r'/register/',
                    r'/contact/',
                    r'/about/',
                    r'/privacy/',
                    r'/terms/',
                    r'/cookie/',
                    r'/disclaimer/',
                )
            ),
            callback='parse_page',
            follow=True
        ),
    )
    
    def parse_page(self, response):
        # Extract article content
        title = response.css('h1::text, .title::text, .article-title::text').get()
        if not title:
            return
            
        # Get all text content
        content = ' '.join(response.css('p::text, article p::text, .content p::text, .article-content p::text').getall())
        
        # Clean content
        content = re.sub(r'\s+', ' ', content).strip()
        
        if not content:
            return
            
        # Extract date if available
        date = response.css('time::attr(datetime), .date::text, .published::text, .article-date::text').get()
        if date:
            try:
                date = datetime.fromisoformat(date.replace('Z', '+00:00'))
            except:
                date = None
        
        # Create article item
        article = {
            'url': response.url,
            'title': title,
            'content': content,
            'domain': response.url.split('/')[2],
            'publish_date': date.isoformat() if date else None,
            'scraped_at': datetime.utcnow().isoformat(),
            'word_count': len(content.split()),
            'is_sme_related': None  # Will be set by the pipeline
        }
        
        yield article 