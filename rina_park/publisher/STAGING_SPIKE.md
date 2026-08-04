# Optional ephemeral media staging spike

This is an optional transport experiment, not the default publisher. The default remains an
Instagram UI package. No tunnel is started by this repository.

Implement `EphemeralProvider` from `staging.py` only after a dedicated test account is ready.
The implementation must:

- expose only the exact files supplied to `expose`; never expose a directory or support listing;
- use a fresh, unguessable route for every approved asset and accept only `GET` and `HEAD`;
- follow no caller-supplied path and reject traversal, symlinks, hardlinks, and real paths outside
  `approved_exports` (the export loader enforces these gates before staging);
- return public HTTPS with a valid certificate and exact MIME and content length;
- apply a short lifetime below Instagram's 24-hour container lifetime;
- log neither URLs, access tokens, local paths, query strings, nor response bodies containing
  credentials;
- terminate and remove every route immediately after publish or any failure;
- stop before any paid tier or transfer charge and fall back to the UI package.

The provider must not be a general file server. A request for `/`, an unknown opaque route, a
directory, a range outside the file, or any non-`GET`/`HEAD` method must fail closed. Publishing
code must use `ApprovedOnlyStaging` as a context manager so teardown occurs on success, timeout,
or reconciliation. A live tunnel and production credentials require separate human approval.
