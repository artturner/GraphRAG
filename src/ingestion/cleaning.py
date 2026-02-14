"""Text cleaning utilities for normalizing document content.

This module provides the TextCleaner class for cleaning and normalizing
text content from various document sources.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CleaningOptions:
    """Configuration options for text cleaning.
    
    Attributes:
        normalize_whitespace: Whether to normalize whitespace characters.
        remove_html: Whether to remove HTML tags.
        normalize_unicode: Whether to normalize unicode characters.
        remove_boilerplate: Whether to remove common boilerplate text.
        preserve_paragraphs: Whether to preserve paragraph breaks.
        lowercase: Whether to convert text to lowercase.
        remove_urls: Whether to remove URLs from text.
        remove_emails: Whether to remove email addresses from text.
    """
    normalize_whitespace: bool = True
    remove_html: bool = True
    normalize_unicode: bool = True
    remove_boilerplate: bool = True
    preserve_paragraphs: bool = True
    lowercase: bool = False
    remove_urls: bool = False
    remove_emails: bool = False


# Common boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    # Copyright notices
    r'Copyright\s*(?:©|\([cC]\))?\s*\d{4}.*?(?:\n|$)',
    r'©\s*\d{4}.*?(?:\n|$)',
    # Common headers
    r'^\s*Cookie\s+Policy\s*$',
    r'^\s*Privacy\s+Policy\s*$',
    r'^\s*Terms\s+of\s+Service\s*$',
    r'^\s*Terms\s+and\s+Conditions\s*$',
    # Navigation elements
    r'^\s*Skip\s+to\s+(?:main\s+)?content\s*$',
    r'^\s*Menu\s*$',
    r'^\s*Navigation\s*$',
    # Social media prompts
    r'^\s*Follow\s+us\s+on\s+.*?$',
    r'^\s*Share\s+(?:this\s+)?(?:article|page|post)\s*$',
    r'^\s*Subscribe\s+(?:to\s+our\s+)?(?:newsletter|mailing\s+list)\s*$',
    # Common footers
    r'^\s*All\s+rights\s+reserved\.?\s*$',
    r'^\s*Last\s+updated:.*?$',
    r'^\s*Page\s+\d+\s+of\s+\d+\s*$',
    # Advertisement placeholders
    r'^\s*Advertisement\s*$',
    r'^\s*AD\s*$',
    r'^\s*Sponsored\s*$',
    # Print/PDF prompts
    r'^\s*Print\s+(?:this\s+)?page\s*$',
    r'^\s*Download\s+PDF\s*$',
    # Login/Register prompts
    r'^\s*(?:Sign\s+in|Log\s+in|Login)\s*$',
    r'^\s*Register\s*$',
    r'^\s*Create\s+(?:an\s+)?account\s*$',
]

# Compile patterns for efficiency
COMPILED_BOILERPLATE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in BOILERPLATE_PATTERNS]


class TextCleaner:
    """Text cleaner for normalizing document content.
    
    This class provides methods for cleaning and normalizing text from
    various document sources. It can handle whitespace normalization,
    HTML tag removal, unicode normalization, and boilerplate removal.
    
    Attributes:
        options: CleaningOptions instance configuring the cleaning behavior.
    
    Example:
        >>> cleaner = TextCleaner()
        >>> dirty = "<p>  Hello   World  </p>"
        >>> clean = cleaner.clean(dirty)
        >>> print(clean)
        Hello World
    """
    
    def __init__(
        self,
        options: Optional[CleaningOptions] = None,
        *,
        normalize_whitespace: bool = True,
        remove_html: bool = True,
        normalize_unicode: bool = True,
        remove_boilerplate: bool = True,
        preserve_paragraphs: bool = True,
        lowercase: bool = False,
        remove_urls: bool = False,
        remove_emails: bool = False,
    ):
        """Initialize the TextCleaner with specified options.
        
        Args:
            options: CleaningOptions instance. If provided, individual
                keyword arguments are ignored.
            normalize_whitespace: Whether to normalize whitespace.
            remove_html: Whether to remove HTML tags.
            normalize_unicode: Whether to normalize unicode characters.
            remove_boilerplate: Whether to remove boilerplate text.
            preserve_paragraphs: Whether to preserve paragraph breaks.
            lowercase: Whether to convert text to lowercase.
            remove_urls: Whether to remove URLs from text.
            remove_emails: Whether to remove email addresses from text.
        """
        if options is not None:
            self.options = options
        else:
            self.options = CleaningOptions(
                normalize_whitespace=normalize_whitespace,
                remove_html=remove_html,
                normalize_unicode=normalize_unicode,
                remove_boilerplate=remove_boilerplate,
                preserve_paragraphs=preserve_paragraphs,
                lowercase=lowercase,
                remove_urls=remove_urls,
                remove_emails=remove_emails,
            )
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.
        
        This method:
        - Replaces multiple spaces with a single space
        - Replaces tabs with spaces
        - Normalizes line endings to \\n
        - Removes leading/trailing whitespace from lines
        - Optionally preserves paragraph breaks (double newlines)
        
        Args:
            text: The text to normalize.
        
        Returns:
            The text with normalized whitespace.
        
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.normalize_whitespace("  Hello   World  ")
            'Hello World'
        """
        if not text:
            return text
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Replace tabs with spaces
        text = text.replace('\t', ' ')
        
        if self.options.preserve_paragraphs:
            # Preserve paragraph breaks (double newlines)
            # First, mark paragraph breaks
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
            # Split by paragraph breaks
            paragraphs = text.split('\n\n')
            
            # Normalize each paragraph
            normalized_paragraphs = []
            for para in paragraphs:
                # Normalize spaces within the paragraph
                para = re.sub(r' +', ' ', para)
                # Remove leading/trailing whitespace from lines
                lines = [line.strip() for line in para.split('\n')]
                # Join lines with single space (or keep as separate lines)
                para = ' '.join(line for line in lines if line)
                if para:
                    normalized_paragraphs.append(para)
            
            # Join paragraphs with double newlines
            text = '\n\n'.join(normalized_paragraphs)
        else:
            # Remove all newlines and normalize spaces
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text
    
    def remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text.
        
        This method removes HTML tags while preserving the text content.
        It also handles:
        - HTML entities (converts to characters)
        - Script and style tags (removes content)
        - HTML comments
        
        Args:
            text: The text to remove HTML tags from.
        
        Returns:
            The text with HTML tags removed.
        
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.remove_html_tags("<p>Hello <b>World</b></p>")
            'Hello World'
        """
        if not text:
            return text
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove script and style tags with content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove all other HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Handle common HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&': '&',
            '<': '<',
            '>': '>',
            '"': '"',
            ''': "'",
            ''': "'",
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '...',
            '&copy;': '©',
            '&reg;': '®',
            '&trade;': '™',
        }
        
        for entity, char in html_entities.items():
            text = text.replace(entity, char)
        
        # Handle numeric entities
        text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
        text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)
        
        return text
    
    def normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters in text.
        
        This method normalizes unicode characters to NFC form and
        optionally replaces certain unicode characters with their
        ASCII equivalents for better compatibility.
        
        Args:
            text: The text to normalize.
        
        Returns:
            The text with normalized unicode characters.
        
        Example:
            >>> cleaner = TextCleaner()
            >>> cleaner.normalize_unicode("café")  # Normalizes to NFC form
            'café'
        """
        if not text:
            return text
        
        # Normalize to NFC form (canonical decomposition followed by composition)
        text = unicodedata.normalize('NFC', text)
        
        # Replace common unicode characters with ASCII equivalents
        unicode_replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...',  # Horizontal ellipsis
            '\u2022': '*',  # Bullet
            '\u00a0': ' ',  # Non-breaking space
            '\u200b': '',   # Zero-width space
            '\u200c': '',   # Zero-width non-joiner
            '\u200d': '',   # Zero-width joiner
            '\ufeff': '',   # Byte order mark
        }
        
        for unicode_char, replacement in unicode_replacements.items():
            text = text.replace(unicode_char, replacement)
        
        # Remove control characters except newlines and tabs
        text = ''.join(
            char for char in text
            if not (unicodedata.category(char) == 'Cc' and char not in '\n\t')
        )
        
        return text
    
    def remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text from documents.
        
        This method removes common headers, footers, and other boilerplate
        text that is typically not relevant to the main content.
        
        Args:
            text: The text to remove boilerplate from.
        
        Returns:
            The text with boilerplate removed.
        
        Example:
            >>> cleaner = TextCleaner()
            >>> text = "Main content\\n\\nCopyright 2024 Example Corp"
            >>> cleaner.remove_boilerplate(text)
            'Main content'
        """
        if not text:
            return text
        
        # Apply each boilerplate pattern
        for pattern in COMPILED_BOILERPLATE:
            text = pattern.sub('', text)
        
        # Remove lines that are just numbers (page numbers)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove lines with just punctuation
        text = re.sub(r'^\s*[\-\=\_\*\#\~\.\,\:\;]+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text.
        
        Args:
            text: The text to remove URLs from.
        
        Returns:
            The text with URLs removed.
        """
        # Match common URL patterns
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        text = re.sub(url_pattern, '', text)
        
        # Match www URLs without protocol
        www_pattern = r'www\.[^\s<>"{}|\\^`\[\]]+'
        text = re.sub(www_pattern, '', text)
        
        return text
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text.
        
        Args:
            text: The text to remove email addresses from.
        
        Returns:
            The text with email addresses removed.
        """
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        text = re.sub(email_pattern, '', text)
        
        return text
    
    def clean(self, text: str) -> str:
        """Apply all cleaning steps to text.
        
        This method applies all enabled cleaning steps in the optimal order:
        1. Remove HTML tags (if enabled)
        2. Normalize unicode (if enabled)
        3. Remove URLs (if enabled)
        4. Remove emails (if enabled)
        5. Remove boilerplate (if enabled)
        6. Normalize whitespace (if enabled)
        7. Convert to lowercase (if enabled)
        
        Args:
            text: The text to clean.
        
        Returns:
            The cleaned text.
        
        Example:
            >>> cleaner = TextCleaner()
            >>> dirty = "<p>  Hello   World  </p>"
            >>> clean = cleaner.clean(dirty)
            >>> print(clean)
            Hello World
        """
        if not text:
            return text
        
        # Step 1: Remove HTML tags
        if self.options.remove_html:
            text = self.remove_html_tags(text)
        
        # Step 2: Normalize unicode
        if self.options.normalize_unicode:
            text = self.normalize_unicode(text)
        
        # Step 3: Remove URLs
        if self.options.remove_urls:
            text = self._remove_urls(text)
        
        # Step 4: Remove emails
        if self.options.remove_emails:
            text = self._remove_emails(text)
        
        # Step 5: Remove boilerplate
        if self.options.remove_boilerplate:
            text = self.remove_boilerplate(text)
        
        # Step 6: Normalize whitespace
        if self.options.normalize_whitespace:
            text = self.normalize_whitespace(text)
        
        # Step 7: Convert to lowercase
        if self.options.lowercase:
            text = text.lower()
        
        return text
    
    def __repr__(self) -> str:
        """Return string representation of the TextCleaner."""
        return (
            f"TextCleaner("
            f"normalize_whitespace={self.options.normalize_whitespace}, "
            f"remove_html={self.options.remove_html}, "
            f"normalize_unicode={self.options.normalize_unicode}, "
            f"remove_boilerplate={self.options.remove_boilerplate})"
        )
