#!/usr/bin/env python3
"""
Enhanced File Organizer - Organize files by type, date, with recursive support.

Features:
- Recursive directory organization (organize subdirectories too)
- Date-based subfolder system (Year/Month/Day structure)
- File size statistics and analysis
- Error handling and logging
- Dry-run mode for safe testing
- Duplicate file handling
"""

import logging
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrganizationMode(Enum):
    """Organization mode selection."""

    BY_TYPE = "type"
    BY_DATE = "date"
    BY_TYPE_AND_DATE = "type_and_date"


# File type categories
FILE_CATEGORIES: dict[str, list[str]] = {
    "Images": [
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".svg",
        ".webp",
        ".ico",
        ".tiff",
    ],
    "Documents": [
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".xlsx",
        ".xls",
        ".ppt",
        ".pptx",
        ".odt",
    ],
    "Videos": [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"],
    "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus"],
    "Code": [
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".cpp",
        ".c",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".sql",
        ".sh",
        ".go",
        ".rs",
    ],
    "Archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".iso"],
    "Executables": [".exe", ".msi", ".app", ".deb", ".rpm", ".apk"],
}


def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def validate_directory(directory: str) -> Path:
    """
    Validate that the provided directory exists and is accessible.

    Args:
        directory: Path to the directory to validate

    Returns:
        Path object if valid

    Raises:
        ValueError: If directory doesn't exist or isn't accessible
    """
    path = Path(directory).expanduser().resolve()

    if not path.exists():
        raise ValueError(f"Directory does not exist: {path}")

    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    try:
        list(path.iterdir())
    except PermissionError as e:
        raise ValueError(f"Permission denied accessing directory: {path}") from e

    return path


def categorize_file(file_extension: str) -> str:
    """
    Determine the category for a file based on its extension.

    Args:
        file_extension: File extension (e.g., '.jpg')

    Returns:
        Category name, or 'Other' if not categorized
    """
    file_extension = file_extension.lower()

    for category, extensions in FILE_CATEGORIES.items():
        if file_extension in extensions:
            return category

    return "Other"


def get_file_date_path(file_path: Path) -> str:
    """
    Get the date-based path for a file.

    Args:
        file_path: Path to the file

    Returns:
        Date path string (e.g., "2026/August/23")
    """
    try:
        timestamp = (
            file_path.stat().st_birthtime
            if hasattr(file_path.stat(), "st_birthtime")
            else file_path.stat().st_mtime
        )
        date = datetime.fromtimestamp(timestamp)
        month_name = date.strftime("%B")
        return f"{date.year}/{month_name}/{date.day:02d}"
    except Exception as e:
        logger.debug(f"Could not get date for {file_path.name}: {e}")
        return "Unknown/Date"


def build_destination_path(
    file_path: Path, base_category_dir: Path, mode: OrganizationMode
) -> Path:
    """
    Build the destination path for a file based on organization mode.

    Args:
        file_path: Source file path
        base_category_dir: Base directory for organization
        mode: Organization mode (by type, date, or both)

    Returns:
        Destination path
    """
    if mode == OrganizationMode.BY_TYPE:
        category = categorize_file(file_path.suffix)
        return base_category_dir / category

    elif mode == OrganizationMode.BY_DATE:
        date_path = get_file_date_path(file_path)
        return base_category_dir / date_path

    elif mode == OrganizationMode.BY_TYPE_AND_DATE:
        category = categorize_file(file_path.suffix)
        date_path = get_file_date_path(file_path)
        return base_category_dir / category / date_path

    return base_category_dir


def get_all_files(directory: Path, recursive: bool = False) -> list[Path]:
    """
    Get all files in a directory, optionally recursive.

    Args:
        directory: Directory to scan
        recursive: If True, include files in subdirectories

    Returns:
        List of file paths
    """
    if recursive:
        return [f for f in directory.rglob("*") if f.is_file()]
    else:
        return [f for f in directory.iterdir() if f.is_file()]


def organize_files(
    directory: str,
    recursive: bool = False,
    mode: OrganizationMode = OrganizationMode.BY_TYPE,
    dry_run: bool = False,
) -> tuple[dict[str, int], list[tuple[str, str]], dict[str, int]]:
    """
    Organize files in the target directory by type and/or date.

    Args:
        directory: Path to the directory to organize
        recursive: If True, organize files in subdirectories too
        mode: Organization mode (type, date, or both)
        dry_run: If True, simulate without moving files

    Returns:
        Tuple of (stats, errors, size_stats)

    Raises:
        ValueError: If directory validation fails
    """
    target_dir = validate_directory(directory)

    # Statistics tracking
    stats = defaultdict(int)
    size_stats = defaultdict(int)  # Size by category
    errors: list[tuple[str, str]] = []

    logger.info(f"Starting file organization in: {target_dir}")
    logger.info(f"Mode: {mode.value} | Recursive: {recursive} | Dry-run: {dry_run}")

    # Get all files
    files = get_all_files(target_dir, recursive=recursive)

    if not files:
        logger.warning("No files found in the directory")
        return dict(stats), errors, dict(size_stats)

    logger.info(f"Found {len(files)} files to organize")

    # Organize each file
    for file_path in files:
        try:
            # Skip organizing files that are already in category directories
            if not recursive and file_path.parent != target_dir:
                continue

            file_size = file_path.stat().st_size

            # Determine destination
            if mode == OrganizationMode.BY_TYPE:
                category = categorize_file(file_path.suffix)
                destination_dir = target_dir / category
            elif mode == OrganizationMode.BY_DATE:
                date_path = get_file_date_path(file_path)
                destination_dir = target_dir / date_path
            else:  # BY_TYPE_AND_DATE
                category = categorize_file(file_path.suffix)
                date_path = get_file_date_path(file_path)
                destination_dir = target_dir / category / date_path

            if not dry_run:
                destination_dir.mkdir(parents=True, exist_ok=True)

                # Handle file name conflicts
                destination = destination_dir / file_path.name
                if destination.exists():
                    counter = 1
                    stem = file_path.stem
                    suffix = file_path.suffix
                    while destination.exists():
                        destination = destination_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                # Move the file
                shutil.move(str(file_path), str(destination))
                logger.debug(
                    f"Moved: {file_path.name} -> {destination.relative_to(target_dir)}"
                )
            else:
                logger.info(
                    f"[DRY RUN] Would move: {file_path.name} -> {destination_dir.relative_to(target_dir)}/"
                )

            # Track statistics
            if mode == OrganizationMode.BY_TYPE:
                category = categorize_file(file_path.suffix)
            elif mode == OrganizationMode.BY_DATE:
                category = get_file_date_path(file_path)
            else:
                category = categorize_file(file_path.suffix)

            stats[category] += 1
            size_stats[category] += file_size

        except Exception as e:
            error_msg = f"Error organizing {file_path.name}: {e!s}"
            logger.error(error_msg)
            errors.append((file_path.name, str(e)))
            stats["errors"] += 1

    return dict(stats), errors, dict(size_stats)


def print_summary(
    stats: dict[str, int],
    size_stats: dict[str, int],
    errors: list[tuple[str, str]],
    dry_run: bool = False,
    mode: OrganizationMode = OrganizationMode.BY_TYPE,
) -> None:
    """
    Print a comprehensive summary of the organization operation.

    Args:
        stats: Dictionary containing movement statistics
        size_stats: Dictionary containing size statistics by category
        errors: List of errors that occurred
        dry_run: Whether this was a dry run
        mode: Organization mode used
    """
    print("\n" + "=" * 70)
    print("FILE ORGANIZATION SUMMARY")
    print("=" * 70)

    if dry_run:
        print("[DRY RUN MODE - No files were actually moved]")
        print()

    total_moved = sum(v for k, v in stats.items() if k != "errors")
    total_size = sum(size_stats.values())

    print(f"Organization Mode: {mode.value}")
    print(f"Total files organized: {total_moved}")
    print(f"Total size: {format_size(total_size)}")

    if stats.get("errors", 0) > 0:
        print(f"Errors encountered: {stats['errors']}")

    print("\n" + "-" * 70)
    print("Breakdown by category:")
    print("-" * 70)
    print(f"{'Category':<20} {'Files':<10} {'Size':<15}")
    print("-" * 70)

    # Sort and display categories
    sorted_categories = sorted(
        [k for k in stats if k != "errors"],
        key=lambda x: stats.get(x, 0),
        reverse=True,
    )

    for category in sorted_categories:
        if stats[category] > 0:
            files_count = stats[category]
            size = size_stats.get(category, 0)
            print(f"{category:<20} {files_count:<10} {format_size(size):<15}")

    print("-" * 70)
    print(f"{'TOTAL':<20} {total_moved:<10} {format_size(total_size):<15}")
    print("-" * 70)

    if errors:
        print("\nErrors encountered:")
        for filename, error in errors:
            print(f"  ✗ {filename}: {error}")

    print("=" * 70 + "\n")


def main() -> None:
    """Main entry point for the enhanced file organizer."""
    if len(sys.argv) < 2:
        print("Usage: python file_organizer_enhanced.py <directory> [OPTIONS]")
        print("\nOptions:")
        print("  --recursive        Organize files in subdirectories too")
        print("  --by-date          Organize by file creation date (Year/Month/Day)")
        print(
            "  --by-type-and-date Organize by type first, then by date (Type/Year/Month/Day)"
        )
        print("  --dry-run          Simulate without moving files")
        print("\nExamples:")
        print("  python file_organizer_enhanced.py ~/Downloads")
        print(
            "  python file_organizer_enhanced.py ~/Downloads --recursive --by-type-and-date --dry-run"
        )
        sys.exit(1)

    directory = sys.argv[1]
    recursive = "--recursive" in sys.argv
    dry_run = "--dry-run" in sys.argv

    # Determine organization mode
    if "--by-type-and-date" in sys.argv:
        mode = OrganizationMode.BY_TYPE_AND_DATE
    elif "--by-date" in sys.argv:
        mode = OrganizationMode.BY_DATE
    else:
        mode = OrganizationMode.BY_TYPE

    try:
        stats, errors, size_stats = organize_files(
            directory, recursive=recursive, mode=mode, dry_run=dry_run
        )
        print_summary(stats, size_stats, errors, dry_run=dry_run, mode=mode)

        if errors and not dry_run:
            sys.exit(1)
        else:
            sys.exit(0)

    except ValueError as e:
        logger.error(f"Invalid directory: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
