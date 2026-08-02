# Third-Party Licensing Boundary

The project license covers this repository's code, not third-party software,
datasets, model weights, or services. Installed Python dependencies retain
their own license terms; the resolved dependency set is recorded in
`uv.lock`.

Detection is not part of the canonical `canine_identity.IdentityEngine` runtime. Detectors supplied
or installed by users, including separately obtained Ultralytics software,
are not project dependencies and remain subject to their own open-source or
commercial terms.

Research localization adapters can use a separately managed Ultralytics
installation. Ultralytics is not included in the package extras because its
AGPL-licensed distribution also conflicts with the headless OpenCV runtime
lane used by the canonical package. Keep that research environment isolated
and review its terms before use.

Datasets and model weights are not distributed by this package. Users must
review and comply with each publisher's access, use, attribution,
redistribution, privacy, and commercial-use terms before supplying data or
weights. The dataset tooling does not grant or replace those permissions.

Any optional user-supplied detector, dataset, or weight artifact is separately
licensed and is the user's responsibility to admit for their intended use.
