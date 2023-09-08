import json
import re
import zipfile
from pathlib import Path

from flask import Flask, request
from markdown import markdown

root = Path("./wiki_bge_small_en_embeddings")
archive = zipfile.ZipFile(root / "data/en/paragraphs.zip", "r")


def get_paragraphs_by_vec_idx(vec_idx):
    chunk_id = vec_idx // 100000
    line_id = vec_idx % 100000
    idmin = max(0, line_id - 20)
    idmax = line_id + 20
    id_set = set(range(idmin, idmax))
    paragraphs = []
    main_paragraph = None
    print("vecid", line_id, "range", id_set)
    with archive.open(
        "enwiki_paragraphs_clean/enwiki_paragraphs_%03d.jsonl" % chunk_id
    ) as f:
        for i, l in enumerate(f):
            if i in id_set:
                paragraph = json.loads(l)
                paragraphs.append(paragraph)
            if i == line_id:
                main_paragraph = paragraph
            if i > idmax:
                break
    return paragraphs, main_paragraph


def render_page(paragraphs, main_paragraph):
    title = main_paragraph["page_title"]
    url = main_paragraph["page_url"]
    page_id = main_paragraph["page_id"]
    page_paragraphs = []
    for p in paragraphs:
        if p["page_id"] == page_id:
            page_paragraphs.append(p)
    text = f"# {title}\n\n"
    last_block_id = None
    last_sublock_id = None
    for p in page_paragraphs:
        titles, block_text = p["text"].split("\n", maxsplit=1)
        block_title = titles.split(". ", maxsplit=1)[1] if ". " in titles else ""
        _, block_id, sublock_id = p["id"].split("_")
        if block_id != last_block_id or sublock_id != last_sublock_id:
            if len(block_title) > 0:
                text += f"\n\n## {block_title}\n\n"
            else:
                text += "\n\n"
        text += block_text
        last_block_id = block_id
        last_sublock_id = sublock_id
    text = re.sub("\n+", "\n\n", text)
    return text, title, url


def get_page(line_id):
    paragraphs, main_paragraph = get_paragraphs_by_vec_idx(line_id)
    return render_page(paragraphs, main_paragraph)


app = Flask(__name__)

from urllib.parse import unquote
@app.route("/")
def hello_world():
    line_id = request.args.get("line_id")
    try:
        line_id = int(line_id)
    except:
        line_id = None
    if line_id is None:
        markdown_page = "Page not found"
        title = "Page not found"
        url = ""
    else:
        markdown_page, title, url = get_page(line_id)
    prefix = """
        <!DOCTYPE html><html lang="en">
        <head>
            <meta charset="utf-8">
            <style type="text/css">.markdown-body { width: 50%}</style>
            <title>""" + title + """</title>
        </head>
        <body><div class="markdown-body">"""
    online_link = f"<div>Online page: <a href=\"{url}\">{unquote(url)}</a></div>" if line_id is not None else ""
    postfix = f"</div>{online_link}</body></html>"
    return prefix + markdown(markdown_page, extensions=['toc']) + postfix


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1380)
