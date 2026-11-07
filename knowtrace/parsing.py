"""Local document parsing with real page and source-line locations."""

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile, BadZipFile

MAX_TEXT_CHARS = 10_000_000
MAX_XML_BYTES = 10 * 1024 * 1024
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


@dataclass(frozen=True)
class Unit:
    text: str
    page: int | None
    line_start: int
    section: str


@dataclass(frozen=True)
class Parsed:
    units: list[Unit]
    parser: str
    page_count: int | None


def _sections(lines: list[str], *, markdown: bool) -> list[Unit]:
    units = []
    heading_stack: list[str] = []
    block: list[str] = []
    start = 1
    section = ""
    fence = ""
    for number, line in enumerate(lines, 1):
        heading = None
        if markdown:
            marker = re.match(r"^\s{0,3}(`{3,}|~{3,})", line)
            if marker:
                character = marker.group(1)[0]
                fence = "" if fence == character else character if not fence else fence
            if not fence:
                heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            if block:
                units.append(Unit("\n".join(block), None, start, section))
            level = len(heading.group(1))
            heading_stack = heading_stack[:level - 1]
            heading_stack.append(heading.group(2).strip())
            section = " / ".join(heading_stack)
            block = [line]
            start = number
        else:
            block.append(line)
    if block:
        units.append(Unit("\n".join(block), None, start, section))
    return units


def _utf8(path: Path, markdown: bool) -> Parsed:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as error:
        raise ValueError("TXT 与 Markdown 文件需要使用 UTF-8 编码") from error
    if "\0" in text:
        raise ValueError("文本含有二进制内容")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("文档文字超过处理上限")
    return Parsed(_sections(text.splitlines(), markdown=markdown),
                  "markdown" if markdown else "text-utf8", None)


def _pdf(path: Path) -> Parsed:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError("PDF 解析依赖 pypdf 尚未安装") from error
    try:
        reader = PdfReader(str(path), strict=False)
        if reader.is_encrypted:
            raise ValueError("请先导出未加密的 PDF 再上传")
        count = len(reader.pages)
        if count > 2000:
            raise ValueError("PDF 超过 2000 页处理上限")
        units = []
        characters = 0
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            characters += len(text)
            if characters > MAX_TEXT_CHARS:
                raise ValueError("PDF 可提取文字超过处理上限")
            units.append(Unit(text, number, 1, ""))
        return Parsed(units, "pdf-pypdf", count)
    except (RuntimeError, ValueError):
        raise
    except Exception as error:
        raise ValueError("无法解析 PDF，请检查文件是否完整") from error


def _xml(archive: ZipFile, name: str, *, optional: bool = False) -> ET.Element | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        if optional:
            return None
        raise ValueError("DOCX 缺少正文")
    if info.file_size > MAX_XML_BYTES:
        raise ValueError("DOCX 正文超过处理上限")
    with archive.open(info) as stream:
        content = stream.read(MAX_XML_BYTES + 1)
    if (len(content) > MAX_XML_BYTES or b"\0" in content
            or b"<!DOCTYPE" in content or b"<!ENTITY" in content):
        raise ValueError("DOCX XML 内容无法处理")
    return ET.fromstring(content)


def _docx(path: Path) -> Parsed:
    try:
        with ZipFile(path) as archive:
            document = _xml(archive, "word/document.xml")
            styles = _xml(archive, "word/styles.xml", optional=True)
        heading_styles = {}
        if styles is not None:
            for style in styles.iter(W + "style"):
                name = style.find(W + "name")
                if name is not None:
                    label = name.get(W + "val", "")
                    match = re.match(r"(?:heading|标题)\s*([1-6])$", label, re.IGNORECASE)
                    if match:
                        heading_styles[style.get(W + "styleId")] = int(match.group(1))
        units = []
        stack: list[str] = []
        block: list[str] = []
        line_number = 1
        start = 1
        section = ""
        characters = 0
        if document is None:
            raise ValueError("DOCX 缺少正文")
        for paragraph in document.iter(W + "p"):
            parts = []
            for element in paragraph.iter():
                if element.tag == W + "t":
                    parts.append(element.text or "")
                elif element.tag == W + "tab":
                    parts.append("\t")
                elif element.tag in (W + "br", W + "cr"):
                    parts.append("\n")
            text = "".join(parts)
            characters += len(text)
            if characters > MAX_TEXT_CHARS:
                raise ValueError("DOCX 文字超过处理上限")
            style = paragraph.find(W + "pPr/" + W + "pStyle")
            style_id = style.get(W + "val", "") if style is not None else ""
            match = re.match(r"Heading([1-6])$", style_id, re.IGNORECASE)
            level = heading_styles.get(style_id) or (int(match.group(1)) if match else None)
            if level and text.strip():
                if block:
                    units.append(Unit("\n".join(block), None, start, section))
                stack = stack[:level - 1]
                stack.append(text.strip())
                section = " / ".join(stack)
                block = []
                start = line_number
            lines = text.split("\n")
            block.extend(lines)
            line_number += len(lines)
        if block:
            units.append(Unit("\n".join(block), None, start, section))
        return Parsed(units, "docx-xml", None)
    except ValueError:
        raise
    except (BadZipFile, ET.ParseError, KeyError, OSError, RuntimeError) as error:
        raise ValueError("无法解析 DOCX，请检查文件是否完整") from error


def parse(path: Path, extension: str) -> Parsed:
    if extension == ".pdf":
        parsed = _pdf(path)
    elif extension == ".docx":
        parsed = _docx(path)
    else:
        parsed = _utf8(path, extension == ".md")
    if not any(unit.text.strip() for unit in parsed.units):
        raise ValueError("未提取到可检索文字；扫描图片 PDF 需要先完成 OCR")
    return parsed


def chunks(parsed: Parsed, *, max_chars: int = 900, overlap_chars: int = 140) -> list[dict]:
    output = []
    for unit in parsed.units:
        lines = []
        for number, line in enumerate(unit.text.splitlines(), unit.line_start):
            pieces = [line[i:i + max_chars] for i in range(0, len(line), max_chars)] or [""]
            lines.extend((piece, number) for piece in pieces)
        current: list[tuple[str, int]] = []
        length = 0

        def emit():
            text = "\n".join(item[0] for item in current).strip("\n")
            if text.strip():
                output.append({"text": text, "page": unit.page, "section": unit.section,
                               "line_start": current[0][1], "line_end": current[-1][1]})

        for line, number in lines:
            if current and length + len(line) + 1 > max_chars:
                emit()
                overlap = []
                overlap_length = 0
                for item in reversed(current[-2:]):
                    if overlap_length + len(item[0]) + 1 > overlap_chars:
                        break
                    overlap.insert(0, item)
                    overlap_length += len(item[0]) + 1
                current = overlap
                length = overlap_length
                if current and length + len(line) + 1 > max_chars:
                    current = []
                    length = 0
            current.append((line, number))
            length += len(line) + 1
        if current:
            emit()
    return output
