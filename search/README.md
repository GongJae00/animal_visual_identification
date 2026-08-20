# Search

In: a query crop or prepared RetrievalQuery, plus a gallery store.

Out: identity-aggregated Match rows from exact available-intersection cosine.

`scoring/` is the query / gallery-key scorer. `matching/` runs extract → score
→ identity max. There is no extra search CLI; the product is
`IdentityEngine.search`.
