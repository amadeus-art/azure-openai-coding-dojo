import re
import os
import urllib.request
import xml.etree.ElementTree as ET
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from typing import Any, Generator, List

import requests
import tiktoken
from bs4 import BeautifulSoup, Doctype, NavigableString, SoupStrainer, Tag
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_random_exponential


class LangchainDocsLoader(BaseLoader):
    """A loader for the Langchain documentation.

    The documentation is available at https://python.langchain.com/.
    """

    _sitemap: str = "https://python.langchain.com/sitemap.xml"
    _filter_urls: List[str] = ["https://python.langchain.com/"]

    def __init__(
        self,
        number_threads: int = os.cpu_count(),
        include_output_cells: bool = True,
        include_links_in_header: bool = False,
    ) -> None:
        """Initialize the loader.

        Args:
            number_threads (int, optional): Number of threads to use
                for parallel processing. Defaults to the number of cpus of the machine.
        """
        self._number_threads = number_threads
        self._include_output_cells = include_output_cells
        self._include_links_in_header = include_links_in_header

    @lru_cache(maxsize=None)
    def load(self) -> List[Document]:
        """Load the documentation.

        Returns:
            List[Document]: A list of documents.
        """

        urls = self._get_urls()
        docs = self._process_urls(urls)
        return docs

    def _get_urls(self) -> List[str]:
        """Get the urls from the sitemap."""
        with urllib.request.urlopen(self._sitemap) as response:
            xml = response.read()

        root = ET.fromstring(xml)

        namespaces = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls = [
            url.text
            for url in root.findall(".//sitemap:loc", namespaces=namespaces)
            if url.text is not None and url.text != "https://python.langchain.com/"
        ]

        return urls

    def _process_urls(self, urls: List[str]) -> List[Document]:
        """Process the urls in parallel."""
        with ThreadPool(self._number_threads) as pool:
            docs = pool.map(self._process_url, urls)
            return docs

    @retry(
        stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=10)
    )
    def _process_url(self, url: str) -> Document:
        """Process a url."""
        r = requests.get(url, allow_redirects=False)
        html = r.text
        metadata = self._metadata_extractor(html, url)
        page_content = self.langchain_docs_extractor(
            html=html,
            include_output_cells=self._include_output_cells,
            # remove the first part of the url
            path_url="/".join(url.split("/")[3:])
            if self._include_links_in_header
            else None,
        )
        return Document(page_content=page_content, metadata=metadata)

    def _metadata_extractor(self, raw_html: str, url: str) -> dict[Any, Any]:
        """Extract metadata from raw html using BeautifulSoup."""
        metadata = {"source": url}

        soup = BeautifulSoup(raw_html, "lxml")
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            if isinstance(description, Tag):
                content = description.get("content", None)
                if isinstance(content, str):
                    metadata["description"] = content
            else:
                metadata["description"] = description.get_text()
        if html := soup.find("html"):
            if isinstance(html, Tag):
                lang = html.get("lang", None)
                if isinstance(lang, str):
                    metadata["language"] = lang

        return metadata

    @staticmethod
    def langchain_docs_extractor(
        html: str,
        include_output_cells: bool,
        path_url: str | None = None,
    ) -> str:
        soup = BeautifulSoup(
            html,
            "lxml",
            parse_only=SoupStrainer(name="article"),
        )

        # Remove all the tags that are not meaningful for the extraction.
        SCAPE_TAGS = ["nav", "footer", "aside", "script", "style"]
        [tag.decompose() for tag in soup.find_all(SCAPE_TAGS)]

        # get url of the page
        def get_text(tag: Tag) -> Generator[str, None, None]:
            for child in tag.children:
                if isinstance(child, Doctype):
                    continue

                if isinstance(child, NavigableString):
                    yield child.get_text()
                elif isinstance(child, Tag):
                    if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                        text = child.get_text(strip=False)

                        if text == "API Reference:":
                            yield f"> **{text}**\n"
                            ul = child.find_next_sibling("ul")
                            if ul is not None and isinstance(ul, Tag):
                                ul.attrs["api_reference"] = "true"
                        else:
                            yield f"{'#' * int(child.name[1:])} "
                            yield from child.get_text(strip=False)

                            if path_url is not None:
                                link = child.find("a")
                                if link is not None:
                                    yield f" [](/{path_url}/{link.get('href')})"
                            yield "\n\n"
                    elif child.name == "a":
                        yield f"[{child.get_text(strip=False)}]({child.get('href')})"
                    elif child.name == "img":
                        yield f"![{child.get('alt', '')}]({child.get('src')})"
                    elif child.name in ["strong", "b"]:
                        yield f"**{child.get_text(strip=False)}**"
                    elif child.name in ["em", "i"]:
                        yield f"_{child.get_text(strip=False)}_"
                    elif child.name == "br":
                        yield "\n"
                    elif child.name == "code":
                        parent = child.find_parent()
                        if parent is not None and parent.name == "pre":
                            classes = parent.attrs.get("class", "")

                            language = next(
                                filter(lambda x: re.match(r"language-\w+", x), classes),
                                None,
                            )
                            if language is None:
                                language = ""
                            else:
                                language = language.split("-")[1]

                            if (
                                language in ["pycon", "text"]
                                and not include_output_cells
                            ):
                                continue

                            lines: list[str] = []
                            for span in child.find_all("span", class_="token-line"):
                                line_content = "".join(
                                    token.get_text() for token in span.find_all("span")
                                )
                                lines.append(line_content)

                            code_content = "\n".join(lines)
                            yield f"```{language}\n{code_content}\n```\n\n"
                        else:
                            yield f"`{child.get_text(strip=False)}`"

                    elif child.name == "p":
                        yield from get_text(child)
                        yield "\n\n"
                    elif child.name == "ul":
                        if "api_reference" in child.attrs:
                            for li in child.find_all("li", recursive=False):
                                yield "> - "
                                yield from get_text(li)
                                yield "\n"
                        else:
                            for li in child.find_all("li", recursive=False):
                                yield "- "
                                yield from get_text(li)
                                yield "\n"
                        yield "\n\n"
                    elif child.name == "ol":
                        for i, li in enumerate(child.find_all("li", recursive=False)):
                            yield f"{i + 1}. "
                            yield from get_text(li)
                            yield "\n\n"
                    elif child.name == "div" and "tabs-container" in child.attrs.get(
                        "class", [""]
                    ):
                        tabs = child.find_all("li", {"role": "tab"})
                        tab_panels = child.find_all("div", {"role": "tabpanel"})
                        for tab, tab_panel in zip(tabs, tab_panels):
                            tab_name = tab.get_text(strip=True)
                            yield f"{tab_name}\n"
                            yield from get_text(tab_panel)
                    elif child.name == "table":
                        thead = child.find("thead")
                        header_exists = isinstance(thead, Tag)
                        if header_exists:
                            headers = thead.find_all("th")
                            if headers:
                                yield "| "
                                yield " | ".join(
                                    header.get_text() for header in headers
                                )
                                yield " |\n"
                                yield "| "
                                yield " | ".join("----" for _ in headers)
                                yield " |\n"

                        tbody = child.find("tbody")
                        tbody_exists = isinstance(tbody, Tag)
                        if tbody_exists:
                            for row in tbody.find_all("tr"):
                                yield "| "
                                yield " | ".join(
                                    cell.get_text(strip=True)
                                    for cell in row.find_all("td")
                                )
                                yield " |\n"

                        yield "\n\n"
                    elif child.name in ["button"]:
                        continue
                    else:
                        yield from get_text(child)

        joined = "".join(get_text(soup))
        return re.sub(r"\n\n+", "\n\n", joined).strip()


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def load_langchain_docs_splitted() -> List[Document]:
    loader = LangchainDocsLoader(include_output_cells=True)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=50,
        length_function=num_tokens_from_string,
    )

    return text_splitter.split_documents(docs)
