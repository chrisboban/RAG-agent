from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ----------------------
# 1. URLs (ADD HERE)
# ----------------------
urls = [
    "https://e42.ai/blog/best-accounts-payable-automation-solution/",
    "https://e42.ai/blog/accounts-payable-in-the-age-of-ai/",
    "https://e42.ai/blog/apacs-digital-journey-from-internet-to-ai-ecosystems/"
]


# ----------------------
# 2. Dynamic Loader (Playwright)
# ----------------------
def load_dynamic_pages(urls):
    docs = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        for url in urls:
            print(f"Loading: {url}")

            page = browser.new_page()
            page.goto(url)

            # Wait until content loads
            page.wait_for_selector("div.elementor-widget-container")

            html = page.content()
            soup = BeautifulSoup(html, "html.parser")

            # Remove obvious noise
            footer = soup.find("div", {"data-elementor-type": "footer"})
            if footer:
                footer.decompose()
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            # Prefer proven body selectors first (full blog text),
            # then fallback to broader article containers.
            elements = soup.select(
                "div.elementor-widget-text-editor p, "
                "div.elementor-widget-text-editor li, "
                "div.elementor-widget-theme-post-content p, "
                "div.elementor-widget-theme-post-content li"
            )
            if not elements:
                article_root = (
                    soup.select_one("article")
                    or soup.select_one("[itemprop='articleBody']")
                    or soup.select_one("main")
                    or soup
                )
                elements = article_root.select("p, li, h2, h3")

            lines = []
            seen = set()
            for el in elements:
                line = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
                if len(line) < 30:
                    continue
                low = line.lower()
                if low in seen:
                    continue
                if any(bad in low for bad in ["cookie", "subscribe", "related posts", "share this"]):
                    continue
                seen.add(low)
                lines.append(line)

            text = "\n".join(lines)
            title = (
                (soup.find("meta", property="og:title") or {}).get("content")
                or (soup.title.string.strip() if soup.title and soup.title.string else url)
            )
            docs.append(Document(page_content=text, metadata={"source": url, "title": title}))

        browser.close()

    return docs


# ----------------------
# 3. Load Documents
# ----------------------
documents = load_dynamic_pages(urls)

print(f"\nLoaded {len(documents)} documents\n")


# ----------------------
# 4. Debug content
# ----------------------
print("\n=== SAMPLE CLEANED CONTENT ===\n")
print(documents[0].page_content[:1000])
print("\n=============================\n")


# ----------------------
# 5. Chunking
# ----------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print(f"\nCreated {len(docs)} chunks\n")


# ----------------------
# 6. Filter (light)
# ----------------------
clean_docs = []

for doc in docs:
    text = doc.page_content.strip()

    if len(text) < 50:
        continue

    clean_docs.append(doc)

print(f"\nFiltered to {len(clean_docs)} strong chunks\n")

# ----------------------
# 6.5 Export chunks for debugging
# ----------------------
chunks_output_path = "chunks.txt"
with open(chunks_output_path, "w", encoding="utf-8") as f:
    for i, doc in enumerate(clean_docs):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "unknown")
        f.write(f"=== CHUNK {i} ===\n")
        f.write(f"Source: {source}\n")
        f.write(f"Title: {title}\n")
        f.write(doc.page_content.strip())
        f.write("\n\n")

print(f"Exported chunks to {chunks_output_path}")


# Safety check
if not clean_docs:
    raise ValueError("No valid documents after filtering. Check extraction logic.")


# ----------------------
# 7. Embeddings
# ----------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ----------------------
# 8. FAISS Index
# ----------------------
vectorstore = FAISS.from_documents(
    clean_docs,
    embeddings
)


# ----------------------
# 9. Save
# ----------------------
vectorstore.save_local("faiss_index")

print("\n✅ FAISS index saved successfully\n")
