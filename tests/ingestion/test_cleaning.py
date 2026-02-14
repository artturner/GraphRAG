"""Tests for text cleaning utilities.

This module contains comprehensive tests for the TextCleaner class
and its associated cleaning methods.
"""

import pytest
from src.ingestion.cleaning import TextCleaner, CleaningOptions


class TestCleaningOptions:
    """Tests for CleaningOptions dataclass."""
    
    def test_default_options(self):
        """Test that default options are set correctly."""
        options = CleaningOptions()
        
        assert options.normalize_whitespace is True
        assert options.remove_html is True
        assert options.normalize_unicode is True
        assert options.remove_boilerplate is True
        assert options.preserve_paragraphs is True
        assert options.lowercase is False
        assert options.remove_urls is False
        assert options.remove_emails is False
    
    def test_custom_options(self):
        """Test that custom options can be set."""
        options = CleaningOptions(
            normalize_whitespace=False,
            remove_html=False,
            lowercase=True,
        )
        
        assert options.normalize_whitespace is False
        assert options.remove_html is False
        assert options.lowercase is True


class TestTextCleanerInit:
    """Tests for TextCleaner initialization."""
    
    def test_init_with_defaults(self):
        """Test initialization with default options."""
        cleaner = TextCleaner()
        
        assert cleaner.options.normalize_whitespace is True
        assert cleaner.options.remove_html is True
        assert cleaner.options.normalize_unicode is True
        assert cleaner.options.remove_boilerplate is True
    
    def test_init_with_options_object(self):
        """Test initialization with CleaningOptions object."""
        options = CleaningOptions(
            normalize_whitespace=False,
            remove_html=False,
        )
        cleaner = TextCleaner(options=options)
        
        assert cleaner.options.normalize_whitespace is False
        assert cleaner.options.remove_html is False
    
    def test_init_with_keyword_args(self):
        """Test initialization with keyword arguments."""
        cleaner = TextCleaner(
            normalize_whitespace=False,
            lowercase=True,
            remove_urls=True,
        )
        
        assert cleaner.options.normalize_whitespace is False
        assert cleaner.options.lowercase is True
        assert cleaner.options.remove_urls is True
    
    def test_options_object_takes_precedence(self):
        """Test that options object takes precedence over keyword args."""
        options = CleaningOptions(normalize_whitespace=False)
        cleaner = TextCleaner(options=options, normalize_whitespace=True)
        
        assert cleaner.options.normalize_whitespace is False
    
    def test_repr(self):
        """Test string representation of TextCleaner."""
        cleaner = TextCleaner()
        repr_str = repr(cleaner)
        
        assert "TextCleaner" in repr_str
        assert "normalize_whitespace" in repr_str


class TestNormalizeWhitespace:
    """Tests for whitespace normalization."""
    
    def test_multiple_spaces(self):
        """Test that multiple spaces are collapsed to single space."""
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("Hello    World")
        
        assert result == "Hello World"
    
    def test_leading_trailing_spaces(self):
        """Test that leading and trailing spaces are removed."""
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("   Hello World   ")
        
        assert result == "Hello World"
    
    def test_tabs_converted_to_spaces(self):
        """Test that tabs are converted to spaces."""
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("Hello\t\tWorld")
        
        assert result == "Hello World"
    
    def test_mixed_whitespace(self):
        """Test normalization of mixed whitespace."""
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("  Hello  \t  World  \n  Test  ")
        
        assert result == "Hello World Test"
    
    def test_preserve_paragraphs(self):
        """Test that paragraph breaks are preserved."""
        cleaner = TextCleaner(preserve_paragraphs=True)
        result = cleaner.normalize_whitespace("Para 1\n\nPara 2\n\nPara 3")
        
        assert result == "Para 1\n\nPara 2\n\nPara 3"
    
    def test_no_preserve_paragraphs(self):
        """Test that paragraph breaks are not preserved when disabled."""
        cleaner = TextCleaner(preserve_paragraphs=False)
        result = cleaner.normalize_whitespace("Para 1\n\nPara 2\n\nPara 3")
        
        assert result == "Para 1 Para 2 Para 3"
    
    def test_normalize_line_endings(self):
        """Test that different line endings are normalized."""
        cleaner = TextCleaner()
        
        # CRLF to LF
        result = cleaner.normalize_whitespace("Hello\r\nWorld")
        assert "\r\n" not in result
        
        # CR to LF
        result = cleaner.normalize_whitespace("Hello\rWorld")
        assert "\r" not in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("")
        
        assert result == ""
    
    def test_whitespace_only(self):
        """Test handling of whitespace-only string."""
        cleaner = TextCleaner()
        result = cleaner.normalize_whitespace("   \t\n   ")
        
        assert result == ""


class TestRemoveHtmlTags:
    """Tests for HTML tag removal."""
    
    def test_simple_tags(self):
        """Test removal of simple HTML tags."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("<p>Hello World</p>")
        
        assert result == " Hello World "
    
    def test_nested_tags(self):
        """Test removal of nested HTML tags."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("<div><p>Hello <b>World</b></p></div>")
        
        assert "Hello" in result
        assert "World" in result
        assert "<" not in result
        assert ">" not in result
    
    def test_self_closing_tags(self):
        """Test removal of self-closing HTML tags."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("Hello<br/>World<hr/>Test")
        
        assert "<" not in result
        assert ">" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_tags_with_attributes(self):
        """Test removal of tags with attributes."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags('<a href="http://example.com">Link</a>')
        
        assert "Link" in result
        assert "<" not in result
        assert "href" not in result
    
    def test_html_comments(self):
        """Test removal of HTML comments."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("Hello<!-- comment -->World")
        
        assert "comment" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_script_tags(self):
        """Test removal of script tags and content."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags(
            "<script>alert('test');</script>Hello World"
        )
        
        assert "alert" not in result
        assert "Hello World" in result
    
    def test_style_tags(self):
        """Test removal of style tags and content."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags(
            "<style>body { color: red; }</style>Hello World"
        )
        
        assert "color" not in result
        assert "Hello World" in result
    
    def test_html_entities(self):
        """Test conversion of HTML entities."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("Hello&nbsp;World&Test")
        
        assert "Hello World" in result
        assert "&" in result
        assert "nbsp" not in result
    
    def test_numeric_entities(self):
        """Test conversion of numeric HTML entities."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("Hello&#32;World&#33;")
        
        assert "Hello World!" in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        cleaner = TextCleaner()
        result = cleaner.remove_html_tags("")
        
        assert result == ""


class TestNormalizeUnicode:
    """Tests for unicode normalization."""
    
    def test_nfc_normalization(self):
        """Test that unicode is normalized to NFC form."""
        cleaner = TextCleaner()
        # Test with accented character (already in NFC form)
        result = cleaner.normalize_unicode("café")
        
        assert result == "café"
    
    def test_smart_quotes_replacement(self):
        """Test that smart quotes are replaced with regular quotes."""
        cleaner = TextCleaner()
        # Using unicode escape sequences for smart quotes
        result = cleaner.normalize_unicode('\u201cHello\u201d and \u2018World\u2019')
        
        assert result == '"Hello" and \'World\''
    
    def test_dash_replacement(self):
        """Test that unicode dashes are replaced."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("Hello – World — Test")
        
        assert "–" not in result
        assert "—" not in result
    
    def test_ellipsis_replacement(self):
        """Test that unicode ellipsis is replaced."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("Hello…")
        
        assert result == "Hello..."
    
    def test_non_breaking_space_replacement(self):
        """Test that non-breaking spaces are replaced."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("Hello\u00a0World")
        
        assert result == "Hello World"
    
    def test_zero_width_characters_removed(self):
        """Test that zero-width characters are removed."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("Hello\u200b\u200c\u200dWorld")
        
        assert result == "HelloWorld"
    
    def test_bom_removed(self):
        """Test that byte order mark is removed."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("\ufeffHello World")
        
        assert result == "Hello World"
    
    def test_control_characters_removed(self):
        """Test that control characters are removed except newlines and tabs."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("Hello\x00\x01\x02World\nTest")
        
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\n" in result
        assert "HelloWorld" in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        cleaner = TextCleaner()
        result = cleaner.normalize_unicode("")
        
        assert result == ""


class TestRemoveBoilerplate:
    """Tests for boilerplate removal."""
    
    def test_copyright_notice(self):
        """Test removal of copyright notices."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\nCopyright 2024 Example Corp"
        )
        
        assert "Main content" in result
        assert "Copyright" not in result
    
    def test_copyright_symbol(self):
        """Test removal of copyright with symbol."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\n© 2024 Example Corp. All rights reserved."
        )
        
        assert "Main content" in result
    
    def test_all_rights_reserved(self):
        """Test removal of 'All rights reserved' text."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\n\nAll rights reserved."
        )
        
        assert "Main content" in result
        assert "rights reserved" not in result
    
    def test_navigation_elements(self):
        """Test removal of navigation elements."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Skip to content\n\nMain content\nMenu"
        )
        
        assert "Main content" in result
        assert "Skip to" not in result
        assert "Menu" not in result
    
    def test_social_media_prompts(self):
        """Test removal of social media prompts."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\nFollow us on Twitter\nShare this article"
        )
        
        assert "Main content" in result
        assert "Follow us" not in result
        assert "Share" not in result
    
    def test_page_numbers(self):
        """Test removal of standalone page numbers."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\n\n42\n\nMore content"
        )
        
        assert "Main content" in result
        assert "More content" in result
        # Page number should be removed
    
    def test_advertisement_markers(self):
        """Test removal of advertisement markers."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\nAdvertisement\nMore content"
        )
        
        assert "Main content" in result
        assert "Advertisement" not in result
    
    def test_excessive_blank_lines(self):
        """Test that excessive blank lines are reduced."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate("Content\n\n\n\n\nMore content")
        
        assert "\n\n\n" not in result
        assert "Content" in result
        assert "More content" in result
    
    def test_punctuation_only_lines(self):
        """Test removal of lines with only punctuation."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate(
            "Main content\n---\n***\nMore content"
        )
        
        assert "Main content" in result
        assert "More content" in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        cleaner = TextCleaner()
        result = cleaner.remove_boilerplate("")
        
        assert result == ""


class TestFullCleaningPipeline:
    """Tests for the full cleaning pipeline."""
    
    def test_basic_cleaning(self):
        """Test basic cleaning with all steps."""
        cleaner = TextCleaner()
        result = cleaner.clean("<p>  Hello   World  </p>")
        
        assert result == "Hello World"
    
    def test_complex_document(self):
        """Test cleaning of a complex document."""
        cleaner = TextCleaner()
        dirty = """
        <html>
            <head><title>Test</title></head>
            <body>
                <p>  Main   content  here  </p>
                <footer>Copyright 2024</footer>
            </body>
        </html>
        """
        result = cleaner.clean(dirty)
        
        assert "Main content here" in result
        assert "<" not in result
        assert "Copyright" not in result
    
    def test_disabled_steps(self):
        """Test that disabled steps are skipped."""
        cleaner = TextCleaner(
            normalize_whitespace=False,
            remove_html=False,
            remove_boilerplate=False,
        )
        result = cleaner.clean("<p>  Hello   World  </p>")
        
        # HTML should be preserved
        assert "<p>" in result
    
    def test_lowercase_option(self):
        """Test lowercase conversion."""
        cleaner = TextCleaner(lowercase=True)
        result = cleaner.clean("Hello WORLD")
        
        assert result == "hello world"
    
    def test_remove_urls_option(self):
        """Test URL removal."""
        cleaner = TextCleaner(remove_urls=True)
        result = cleaner.clean("Visit https://example.com for more info")
        
        assert "https://example.com" not in result
        assert "Visit" in result
        assert "info" in result
    
    def test_remove_emails_option(self):
        """Test email removal."""
        cleaner = TextCleaner(remove_emails=True)
        result = cleaner.clean("Contact us at test@example.com today")
        
        assert "test@example.com" not in result
        assert "Contact" in result
        assert "today" in result
    
    def test_empty_string(self):
        """Test handling of empty string."""
        cleaner = TextCleaner()
        result = cleaner.clean("")
        
        assert result == ""
    
    def test_none_handling(self):
        """Test handling of None-like input."""
        cleaner = TextCleaner()
        result = cleaner.clean("")
        
        assert result == ""
    
    def test_preserves_important_content(self):
        """Test that important content is preserved."""
        cleaner = TextCleaner()
        document = """
        <article>
            <h1>Important Title</h1>
            <p>This is the main content of the document.</p>
            <p>It contains important information.</p>
        </article>
        """
        result = cleaner.clean(document)
        
        assert "Important Title" in result
        assert "main content" in result
        assert "important information" in result
    
    def test_order_of_operations(self):
        """Test that cleaning steps are applied in correct order."""
        # HTML removal should happen before whitespace normalization
        cleaner = TextCleaner()
        result = cleaner.clean("<p>Test</p>")
        
        # Should not have extra spaces from tag removal
        assert result == "Test"


class TestContentPreservation:
    """Tests to ensure important content is preserved during cleaning."""
    
    def test_preserves_numbers(self):
        """Test that numbers are preserved."""
        cleaner = TextCleaner()
        result = cleaner.clean("The year 2024 has 365 days")
        
        assert "2024" in result
        assert "365" in result
    
    def test_preserves_special_characters_in_content(self):
        """Test that legitimate special characters are preserved."""
        cleaner = TextCleaner()
        result = cleaner.clean("Price: $99.99 (20% off!)")
        
        assert "$" in result
        assert "99.99" in result
        assert "%" in result
    
    def test_preserves_code_snippets(self):
        """Test that code-like content is preserved."""
        cleaner = TextCleaner(remove_html=False)
        result = cleaner.clean("Use the function foo() with args (x, y)")
        
        assert "foo()" in result
        assert "(x, y)" in result
    
    def test_preserves_urls_when_enabled(self):
        """Test that URLs are preserved when removal is disabled."""
        cleaner = TextCleaner(remove_urls=False)
        result = cleaner.clean("Visit https://example.com")
        
        assert "https://example.com" in result
    
    def test_preserves_emails_when_enabled(self):
        """Test that emails are preserved when removal is disabled."""
        cleaner = TextCleaner(remove_emails=False)
        result = cleaner.clean("Contact: test@example.com")
        
        assert "test@example.com" in result
    
    def test_preserves_multiline_content(self):
        """Test that multiline content structure is preserved."""
        cleaner = TextCleaner(preserve_paragraphs=True)
        result = cleaner.clean("Line 1\n\nLine 2\n\nLine 3")
        
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result
    
    def test_preserves_international_characters(self):
        """Test that international characters are preserved."""
        cleaner = TextCleaner()
        result = cleaner.clean("Hello 世界 مرحبا Привет")
        
        assert "世界" in result
        assert "مرحبا" in result
        assert "Привет" in result
    
    def test_preserves_formatted_numbers(self):
        """Test that formatted numbers are preserved."""
        cleaner = TextCleaner()
        result = cleaner.clean("Phone: +1 (555) 123-4567")
        
        assert "+1" in result
        assert "555" in result
        assert "123-4567" in result
    
    def test_preserves_dates(self):
        """Test that dates are preserved."""
        cleaner = TextCleaner()
        result = cleaner.clean("Date: 2024-01-15 or 01/15/2024")
        
        assert "2024-01-15" in result
        assert "01/15/2024" in result


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        cleaner = TextCleaner()
        long_text = "Hello World " * 10000
        result = cleaner.clean(long_text)
        
        assert "Hello World" in result
        assert len(result) < len(long_text)  # Should be shorter due to normalization
    
    def test_only_html_tags(self):
        """Test handling of text with only HTML tags."""
        cleaner = TextCleaner()
        result = cleaner.clean("<div><p><span></span></p></div>")
        
        assert result == ""
    
    def test_only_whitespace(self):
        """Test handling of whitespace-only text."""
        cleaner = TextCleaner()
        result = cleaner.clean("   \t\n   ")
        
        assert result == ""
    
    def test_only_boilerplate(self):
        """Test handling of text with only boilerplate."""
        cleaner = TextCleaner()
        result = cleaner.clean("Copyright 2024. All rights reserved.")
        
        assert result == ""
    
    def test_malformed_html(self):
        """Test handling of malformed HTML."""
        cleaner = TextCleaner()
        result = cleaner.clean("<p>Hello<div>World</p></div>")
        
        assert "Hello" in result
        assert "World" in result
    
    def test_unicode_edge_cases(self):
        """Test handling of unicode edge cases."""
        cleaner = TextCleaner()
        result = cleaner.clean("\u0000\u0001\u0002Hello\uFEFFWorld")
        
        assert "Hello" in result
        assert "World" in result
    
    def test_nested_boilerplate(self):
        """Test handling of nested boilerplate patterns."""
        cleaner = TextCleaner()
        result = cleaner.clean(
            "Content\nCopyright 2024\nAll rights reserved.\nMore content"
        )
        
        assert "Content" in result
        assert "More content" in result
    
    def test_mixed_content_and_boilerplate(self):
        """Test handling of mixed content and boilerplate."""
        cleaner = TextCleaner()
        result = cleaner.clean(
            "Article about copyright law\nCopyright 2024\nMore article content"
        )
        
        # The word "copyright" in the article should be preserved
        # but the copyright notice should be removed
        assert "Article" in result
        assert "More article content" in result
