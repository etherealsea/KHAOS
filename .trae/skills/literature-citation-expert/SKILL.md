---
name: "literature-citation-expert"
description: "Expert in academic literature research, verification, and citation formatting. Invoke when user asks to find papers, verify citations, or format references."
---

# Literature Citation Expert

This skill assists with academic literature research, citation verification, and reference formatting.

## Capabilities

1.  **Literature Search**: Finds relevant academic papers based on keywords or topics.
2.  **Citation Verification**: Verifies the accuracy of citations (authors, title, year, venue) to ensure they are real and correct.
3.  **Reference Formatting**: Formats references in standard academic styles (APA, IEEE, BibTeX, etc.).
4.  **Contextual Recommendation**: Suggests relevant literature to support specific arguments or research gaps.

## Usage Guidelines

-   **Verification**: When a user provides a list of references, use search tools to verify each entry's metadata. Correct any errors in author names, titles, or publication details.
-   **Formatting**: When asked to format references, ensure strict adherence to the requested style guide.
-   **Search**: When asked to find papers, provide high-quality, relevant results with brief summaries of why they are relevant.

## Prompts & Techniques

-   **"Verify these references"**: Check each reference against reliable databases (via search) and correct errors.
-   **"Find papers on X"**: Search for key papers on topic X, prioritizing recent and highly cited works.
-   **"Format as IEEE"**: Convert the provided references into standard IEEE format.

## Example Interaction

**User**: "Check this citation: 'Smith (2020) on Deep Learning'."

**Response**:
"I have verified the citation.
**Correction**: The correct title is 'Deep Learning for Time Series', published in *NeurIPS 2020*.
**Formatted (APA)**: Smith, J. (2020). Deep learning for time series. *Advances in Neural Information Processing Systems*, 33, 1234-1245."
