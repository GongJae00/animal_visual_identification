"""GenID and ReID stage: gallery persistence and exact cosine retrieval.

Enroll writes GalleryKey / GalleryValue rows. Search scores a RetrievalQuery
with available-intersection weighted cosine. QKV names are roles, not an
attention mechanism. The public facade is ``runtime.IdentityEngine``.
"""
