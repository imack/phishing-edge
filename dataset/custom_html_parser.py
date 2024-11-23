import html2text
from bs4 import BeautifulSoup
from urllib.parse import urlparse

class CustomHTML2Text(html2text.HTML2Text):
    def handle_meta_link_title(self, soup):
        retained_tags = []
        seen_domains = set()

        # Extract the title
        title_tag = soup.find("title")
        if title_tag:
            retained_tags.append(f"Title: {title_tag.string.strip()}")

        # Extract meta tags and flatten them
        for meta in soup.find_all("meta"):
            meta_attrs = [f"{key} {value}" for key, value in meta.attrs.items()]
            retained_tags.append("Meta " + " ".join(meta_attrs))

        # Extract link tags and format unique domains
        for link in soup.find_all("link"):
            href = link.get("href")
            if href:
                domain = urlparse(href).netloc or "unknown"
                if domain not in seen_domains:
                    seen_domains.add(domain)
                    relation = link.get("rel", ["unknown"])[0]
                    retained_tags.append(f"Link {domain} {relation}")

        return "\n".join(retained_tags)

    def handle(self, html):
        soup = BeautifulSoup(html, "html.parser")
        meta_link_title_tags = self.handle_meta_link_title(soup)
        text_content = super().handle(html)
        return meta_link_title_tags + "\n\n" + text_content
