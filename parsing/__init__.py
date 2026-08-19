"""Parsing stage: frozen detection, segmentation, ROI, and crop materialization.

Full-animal runtime lives in ``parsing.full_segment``. Types live in
``parsing.types``. ``runtime.IdentityEngine`` does not import this
package; it still accepts caller-provided crops.
"""
