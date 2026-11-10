# Superseded run — kept for transparency

This coverage run used the first version of `strip_cover_page`, which cut
everything before the first `Item x.xx` header. That also removed
"Explanatory / Introductory Note" paragraphs that some 8-Ks place between the
cover page and Item 1.01 (E1-054's evidence sits there), so `structured` and
`keyword` lost coverage on the long group.

Fixed the same day: the cover page is now cut after the last
"emerging growth company" check-box line (checked on all 34 dev 8-K main
documents: no gold evidence before the cut). See the next coverage run.
