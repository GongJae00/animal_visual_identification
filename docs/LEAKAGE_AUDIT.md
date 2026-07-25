# Structural Leakage Audit

## Episode-weighted association

Adjacent frames and fragmented tracks are not independent observations.
Structural shortcut risk is therefore computed after collapsing records to one
event per `(occupancy episode, registered dog)`.

For dog identity \(X\) and camera or cage domain \(Y\), the audit reports:

\[
I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
\]

\[
\operatorname{NMI}(X,Y)=\frac{I(X;Y)}{\sqrt{H(X)H(Y)}}.
\]

It also reports:

- domain→dog majority accuracy: how well identity can be guessed from only a
  camera or cage label;
- global-majority dog accuracy: the no-domain class-prior control;
- dog→domain concentration: the fraction of identity episodes retained by the
  most common domain for each dog.

NMI is `null` when fewer than two dogs or two domains make it undefined.
No universal pass threshold is frozen before the real cohort distribution is
known.

## Interpretation boundary

Low metadata association does not prove background invariance. High association
proves a shortcut opportunity, not that a visual model used it. The mandatory
later controls remain:

- background-only identity classification;
- cage-only and accessory-only classification;
- body-blurred inputs;
- random background replacement;
- cross-cage and cross-camera tests.

Association statistics are episode-weighted diagnostics and must not be
presented as recognition accuracy.
