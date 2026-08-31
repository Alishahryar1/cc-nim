"""Generated local file storage for Chat Sessions attachments."""

import os
import shutil
import uuid
import zipfile
from functools import partial
from pathlib import Path, PurePosixPath
from typing import BinaryIO

import anyio.to_thread
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from free_claude_code.application.chat.models import (
    MAX_CHAT_ATTACHMENT_BYTES,
    MAX_CHAT_ATTACHMENT_EXTRACTED_CHARACTERS,
    ChatAttachment,
    ChatAttachmentContent,
    ChatAttachmentFileInfo,
    ChatAttachmentKind,
    ChatAttachmentMaterial,
    ChatDocumentAttachment,
    ChatImageAttachment,
    ChatPayloadTooLargeError,
    ChatUnavailableError,
    ChatUnsupportedAttachmentError,
    ChatValidationError,
)

_ATTACHMENTS_DIRNAME = "attachments"
_TEMP_DIRNAME = "tmp"
_ORIGINAL_FILENAME = "original"
_EXTRACTED_FILENAME = "extracted.txt"
_COPY_CHUNK_BYTES = 64 * 1024
_MAX_PDF_PAGES = 400
_MAX_DOCX_ENTRIES = 1_000
_MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
_GENERIC_MEDIA_TYPES = frozenset(
    {"", "application/octet-stream", "binary/octet-stream"}
)
_TEXT_EXTENSIONS = frozenset({".txt"})
_MARKDOWN_EXTENSIONS = frozenset({".md", ".markdown"})
_SUPPORTED_EXTENSIONS = {
    ".jpg": ChatAttachmentKind.IMAGE,
    ".jpeg": ChatAttachmentKind.IMAGE,
    ".png": ChatAttachmentKind.IMAGE,
    ".gif": ChatAttachmentKind.IMAGE,
    ".webp": ChatAttachmentKind.IMAGE,
    ".txt": ChatAttachmentKind.TEXT,
    ".md": ChatAttachmentKind.MARKDOWN,
    ".markdown": ChatAttachmentKind.MARKDOWN,
    ".pdf": ChatAttachmentKind.PDF,
    ".docx": ChatAttachmentKind.DOCX,
}
_DECLARED_MEDIA_TYPES = {
    "image/jpeg": frozenset({"image/jpeg", "image/jpg"}),
    "image/png": frozenset({"image/png"}),
    "image/gif": frozenset({"image/gif"}),
    "image/webp": frozenset({"image/webp"}),
    "text/plain": frozenset({"text/plain"}),
    "text/markdown": frozenset({"text/markdown", "text/x-markdown"}),
    "application/pdf": frozenset({"application/pdf"}),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
        frozenset(
            {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/zip",
                "application/x-zip-compressed",
            }
        )
    ),
}


class LocalChatAttachmentFiles:
    """Own attachment paths, validation, extraction, and orphan cleanup."""

    def __init__(self, chat_state_dir: Path) -> None:
        self._root = chat_state_dir / _ATTACHMENTS_DIRNAME
        self._temp = chat_state_dir / _TEMP_DIRNAME

    async def start(self, owners: tuple[tuple[str, str], ...]) -> None:
        await anyio.to_thread.run_sync(self._start_sync, owners)

    async def store_upload(
        self,
        *,
        session_id: str,
        attachment_id: str,
        filename: str,
        declared_media_type: str | None,
        source: BinaryIO,
    ) -> ChatAttachmentFileInfo:
        return await anyio.to_thread.run_sync(
            partial(
                self._store_upload_sync,
                session_id=session_id,
                attachment_id=attachment_id,
                filename=filename,
                declared_media_type=declared_media_type,
                source=source,
            )
        )

    async def materialize(
        self, attachments: tuple[ChatAttachment, ...]
    ) -> tuple[ChatAttachmentMaterial, ...]:
        return await anyio.to_thread.run_sync(self._materialize_sync, attachments)

    async def content(self, attachment: ChatAttachment) -> ChatAttachmentContent:
        return await anyio.to_thread.run_sync(self._content_sync, attachment)

    async def available_ids(
        self, attachments: tuple[ChatAttachment, ...]
    ) -> frozenset[str]:
        return await anyio.to_thread.run_sync(self._available_ids_sync, attachments)

    async def delete_attachment(self, attachment: ChatAttachment) -> None:
        await anyio.to_thread.run_sync(
            partial(
                self._remove_tree,
                self._attachment_dir(attachment.session_id, attachment.id),
            )
        )

    async def delete_session(self, session_id: str) -> None:
        await anyio.to_thread.run_sync(
            partial(self._remove_tree, self._session_dir(session_id))
        )

    def _start_sync(self, owners: tuple[tuple[str, str], ...]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._temp.mkdir(parents=True, exist_ok=True)
        _owner_only(self._root, 0o700)
        _owner_only(self._temp, 0o700)
        valid = frozenset(owners)
        for child in tuple(self._temp.iterdir()):
            if _is_uuid(child.name):
                self._remove_tree(child)
        for session_dir in tuple(self._root.iterdir()):
            if not session_dir.is_dir() or not _is_uuid(session_dir.name):
                continue
            for attachment_dir in tuple(session_dir.iterdir()):
                if not attachment_dir.is_dir() or not _is_uuid(attachment_dir.name):
                    continue
                if (session_dir.name, attachment_dir.name) not in valid:
                    self._remove_tree(attachment_dir)
            if not any(session_dir.iterdir()):
                session_dir.rmdir()

    def _store_upload_sync(
        self,
        *,
        session_id: str,
        attachment_id: str,
        filename: str,
        declared_media_type: str | None,
        source: BinaryIO,
    ) -> ChatAttachmentFileInfo:
        _require_uuid(session_id, "session")
        _require_uuid(attachment_id, "attachment")
        self._root.mkdir(parents=True, exist_ok=True)
        self._temp.mkdir(parents=True, exist_ok=True)
        _owner_only(self._root, 0o700)
        _owner_only(self._temp, 0o700)
        temporary = self._temp / attachment_id
        final = self._attachment_dir(session_id, attachment_id)
        if temporary.exists() or final.exists():
            raise ChatUnavailableError("Could not allocate attachment storage.")
        temporary.mkdir(mode=0o700)
        _owner_only(temporary, 0o700)
        original = temporary / _ORIGINAL_FILENAME
        try:
            byte_size = _copy_bounded(source, original)
            kind, media_type, extracted = _inspect_attachment(
                original,
                filename=filename,
                declared_media_type=declared_media_type,
            )
            extracted_characters: int | None = None
            if extracted is not None:
                extracted_characters = len(extracted)
                if extracted_characters > MAX_CHAT_ATTACHMENT_EXTRACTED_CHARACTERS:
                    raise ChatPayloadTooLargeError(
                        "Attachment text exceeds 1,000,000 characters."
                    )
                extracted_path = temporary / _EXTRACTED_FILENAME
                extracted_path.write_text(extracted, encoding="utf-8", newline="\n")
                _owner_only(extracted_path, 0o600)
            _owner_only(original, 0o600)
            session_dir = self._session_dir(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            _owner_only(session_dir, 0o700)
            os.replace(temporary, final)
            _owner_only(final, 0o700)
            return ChatAttachmentFileInfo(
                kind=kind,
                media_type=media_type,
                byte_size=byte_size,
                extracted_characters=extracted_characters,
            )
        except ChatPayloadTooLargeError, ChatUnsupportedAttachmentError:
            self._remove_tree(temporary)
            raise
        except OSError as exc:
            self._remove_tree(temporary)
            raise ChatUnavailableError("Could not store the attachment.") from exc
        except Exception as exc:
            self._remove_tree(temporary)
            raise ChatUnsupportedAttachmentError(
                "The attachment could not be read safely."
            ) from exc

    def _materialize_sync(
        self, attachments: tuple[ChatAttachment, ...]
    ) -> tuple[ChatAttachmentMaterial, ...]:
        materials: list[ChatAttachmentMaterial] = []
        for attachment in attachments:
            directory = self._attachment_dir(attachment.session_id, attachment.id)
            if attachment.kind is ChatAttachmentKind.IMAGE:
                data = _read_original(directory / _ORIGINAL_FILENAME, attachment)
                materials.append(ChatImageAttachment(attachment=attachment, data=data))
                continue
            extracted_path = directory / _EXTRACTED_FILENAME
            try:
                text = extracted_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise _unavailable_attachment(attachment) from exc
            if (
                attachment.extracted_characters is None
                or len(text) != attachment.extracted_characters
            ):
                raise _unavailable_attachment(attachment)
            materials.append(ChatDocumentAttachment(attachment=attachment, text=text))
        return tuple(materials)

    def _content_sync(self, attachment: ChatAttachment) -> ChatAttachmentContent:
        data = _read_original(
            self._attachment_dir(attachment.session_id, attachment.id)
            / _ORIGINAL_FILENAME,
            attachment,
        )
        return ChatAttachmentContent(attachment=attachment, data=data)

    def _available_ids_sync(
        self, attachments: tuple[ChatAttachment, ...]
    ) -> frozenset[str]:
        available: set[str] = set()
        for attachment in attachments:
            directory = self._attachment_dir(attachment.session_id, attachment.id)
            original = directory / _ORIGINAL_FILENAME
            if not original.is_file():
                continue
            try:
                if original.stat().st_size != attachment.byte_size:
                    continue
            except OSError:
                continue
            if attachment.kind is not ChatAttachmentKind.IMAGE:
                extracted = directory / _EXTRACTED_FILENAME
                if not extracted.is_file():
                    continue
            available.add(attachment.id)
        return frozenset(available)

    def _session_dir(self, session_id: str) -> Path:
        _require_uuid(session_id, "session")
        return self._root / session_id

    def _attachment_dir(self, session_id: str, attachment_id: str) -> Path:
        _require_uuid(attachment_id, "attachment")
        return self._session_dir(session_id) / attachment_id

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)


def _copy_bounded(source: BinaryIO, target: Path) -> int:
    total = 0
    source.seek(0)
    with target.open("xb") as destination:
        while True:
            chunk = source.read(_COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_CHAT_ATTACHMENT_BYTES:
                raise ChatPayloadTooLargeError(
                    "Attachments may be at most 10 MiB each."
                )
            destination.write(chunk)
    if total <= 0:
        raise ChatUnsupportedAttachmentError("The attachment is empty.")
    return total


def _inspect_attachment(
    path: Path,
    *,
    filename: str,
    declared_media_type: str | None,
) -> tuple[ChatAttachmentKind, str, str | None]:
    suffix = Path(filename).suffix.casefold()
    with path.open("rb") as source:
        header = source.read(16)

    kind: ChatAttachmentKind
    media_type: str
    extracted: str | None = None
    if header.startswith(b"\xff\xd8\xff"):
        kind, media_type = ChatAttachmentKind.IMAGE, "image/jpeg"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        kind, media_type = ChatAttachmentKind.IMAGE, "image/png"
    elif header.startswith((b"GIF87a", b"GIF89a")):
        kind, media_type = ChatAttachmentKind.IMAGE, "image/gif"
    elif len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        kind, media_type = ChatAttachmentKind.IMAGE, "image/webp"
    elif header.startswith(b"%PDF-"):
        kind, media_type = ChatAttachmentKind.PDF, "application/pdf"
        extracted = _extract_pdf(path)
    elif header.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        kind = ChatAttachmentKind.DOCX
        media_type = (
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        extracted = _extract_docx(path)
    elif suffix in _TEXT_EXTENSIONS | _MARKDOWN_EXTENSIONS:
        kind = (
            ChatAttachmentKind.MARKDOWN
            if suffix in _MARKDOWN_EXTENSIONS
            else ChatAttachmentKind.TEXT
        )
        media_type = (
            "text/markdown" if kind is ChatAttachmentKind.MARKDOWN else "text/plain"
        )
        extracted = _read_utf8_text(path)
    else:
        raise ChatUnsupportedAttachmentError(
            "Supported attachments are JPEG, PNG, GIF, WebP, TXT, Markdown, PDF, and DOCX."
        )

    _validate_extension(kind, media_type, suffix)
    _validate_declared_media_type(media_type, declared_media_type)
    if extracted is not None and not extracted.strip():
        raise ChatUnsupportedAttachmentError(
            "The document contains no readable text. OCR is not supported."
        )
    return kind, media_type, extracted


def _validate_extension(kind: ChatAttachmentKind, media_type: str, suffix: str) -> None:
    if not suffix:
        return
    expected_kind = _SUPPORTED_EXTENSIONS.get(suffix)
    if expected_kind is None or expected_kind is not kind:
        raise ChatUnsupportedAttachmentError(
            "The filename extension does not match the attachment content."
        )
    if kind is ChatAttachmentKind.IMAGE:
        allowed = {
            "image/jpeg": frozenset({".jpg", ".jpeg"}),
            "image/png": frozenset({".png"}),
            "image/gif": frozenset({".gif"}),
            "image/webp": frozenset({".webp"}),
        }[media_type]
        if suffix not in allowed:
            raise ChatUnsupportedAttachmentError(
                "The filename extension does not match the image content."
            )


def _validate_declared_media_type(
    verified_media_type: str, declared_media_type: str | None
) -> None:
    declared = (declared_media_type or "").partition(";")[0].strip().casefold()
    if declared in _GENERIC_MEDIA_TYPES:
        return
    if declared not in _DECLARED_MEDIA_TYPES[verified_media_type]:
        raise ChatUnsupportedAttachmentError(
            "The declared media type does not match the attachment content."
        )


def _read_utf8_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ChatUnsupportedAttachmentError(
            "Text attachments must use UTF-8 encoding."
        ) from exc
    return _normalize_text(text)


def _extract_pdf(path: Path) -> str:
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ChatUnsupportedAttachmentError(
                "Encrypted PDF files are not supported."
            )
        if len(reader.pages) > _MAX_PDF_PAGES:
            raise ChatPayloadTooLargeError("PDF files may contain at most 400 pages.")
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    except ChatPayloadTooLargeError, ChatUnsupportedAttachmentError:
        raise
    except Exception as exc:
        raise ChatUnsupportedAttachmentError("The PDF file is malformed.") from exc
    return _normalize_text(text)


def _extract_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > _MAX_DOCX_ENTRIES:
                raise ChatPayloadTooLargeError(
                    "DOCX files may contain at most 1,000 ZIP entries."
                )
            total_uncompressed = 0
            names: set[str] = set()
            for entry in entries:
                _validate_zip_entry(entry)
                total_uncompressed += entry.file_size
                if total_uncompressed > _MAX_DOCX_UNCOMPRESSED_BYTES:
                    raise ChatPayloadTooLargeError(
                        "DOCX uncompressed content may be at most 50 MiB."
                    )
                names.add(entry.filename)
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise ChatUnsupportedAttachmentError(
                    "The DOCX file is missing required document data."
                )
        document = Document(str(path))
        parts: list[str] = []
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                if item.text:
                    parts.append(item.text)
            elif isinstance(item, Table):
                for row in item.rows:
                    cells = [cell.text for cell in row.cells]
                    if any(cell.strip() for cell in cells):
                        parts.append("\t".join(cells))
        return _normalize_text("\n".join(parts))
    except ChatPayloadTooLargeError, ChatUnsupportedAttachmentError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ChatUnsupportedAttachmentError("The DOCX file is malformed.") from exc


def _validate_zip_entry(entry: zipfile.ZipInfo) -> None:
    if entry.flag_bits & 0x1:
        raise ChatUnsupportedAttachmentError("Encrypted DOCX files are not supported.")
    name = entry.filename
    pure = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or name.startswith("/")
        or pure.is_absolute()
        or ".." in pure.parts
    ):
        raise ChatUnsupportedAttachmentError("The DOCX file contains an unsafe path.")


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_original(path: Path, attachment: ChatAttachment) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _unavailable_attachment(attachment) from exc
    if len(data) != attachment.byte_size or len(data) > MAX_CHAT_ATTACHMENT_BYTES:
        raise _unavailable_attachment(attachment)
    return data


def _unavailable_attachment(attachment: ChatAttachment) -> ChatValidationError:
    action = (
        "Remove it or delete the chat."
        if attachment.turn_id is None
        else "Delete the chat to remove its reference."
    )
    return ChatValidationError(
        f"Attachment {attachment.filename!r} is unavailable. {action}"
    )


def _require_uuid(value: str, label: str) -> None:
    if not _is_uuid(value):
        raise ChatValidationError(f"Invalid {label} ID.")


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError, AttributeError:
        return False


def _owner_only(path: Path, mode: int) -> None:
    if os.name != "nt" and path.exists():
        path.chmod(mode)
