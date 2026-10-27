# LANL repository mirror

[lanl/DeepReservoir-Navajo](https://github.com/lanl/DeepReservoir-Navajo) is an
automated one-way mirror of
[VeinsOfTheEarth/DeepReservoir-Navajo](https://github.com/VeinsOfTheEarth/DeepReservoir-Navajo).
Develop and contribute upstream, not directly in the LANL mirror.

## Synchronization

The upstream `workflows/mirror.yml` runs on branch/tag pushes, creation, and
deletion, and supports manual dispatch from **Actions > Mirror to LANL**.
It fetches all upstream branches/tags into a fresh bare repository and pushes
them atomically to LANL. Deleted refs are pruned; mirror-only changes are
intentionally overwritten. The `main` guard prevents a misconfigured or empty
source from deleting mirror refs. Update that guard and the destination default
branch deliberately if upstream `main` is renamed.

The mirror includes tracked files and Git history, not GitHub issues, pull
requests, release descriptions/download attachments, or repository settings.
Those services remain upstream. GitHub Actions and issues are disabled in the
LANL mirror to avoid duplicate workflows and contribution threads. Repository
descriptions are managed separately and are not changed by synchronization.

## Credentials and trust

The dedicated write-enabled deploy key belongs only to `lanl/DeepReservoir-Navajo`.
Its private half is the upstream Actions secret
`LANL_DEEPRESERVOIR_NAVAJO_MIRROR_SSH_KEY`. Public upstream data is fetched via
HTTPS, so no second key is needed. The key is not attached to a personal account
and has no automatic expiry. No local workstation or scheduler is required.

Only trusted upstream writers should have direct push access: branch workflows
can access the secret and therefore control the mirror. Fork pull requests do
not trigger this workflow. Rotate the credential if that trust changes; never
commit the private key. SSH host verification remains enabled using host keys
from GitHub's HTTPS API.

## Verification and recovery

Compare all branches/tags when checking synchronization:

```bash
git ls-remote --heads --tags https://github.com/VeinsOfTheEarth/DeepReservoir-Navajo.git
git ls-remote --heads --tags https://github.com/lanl/DeepReservoir-Navajo.git
```

Matching ref names and object IDs confirm the same Git history. Check the
upstream mirror run if they differ, resolve permissions/key/ruleset failures,
then use **Run workflow** to retry. The job does not suppress failures.

GitHub can suppress ref events for bulk operations and changes made using
`GITHUB_TOKEN`. Explicitly dispatch this workflow when introducing such
automation; a later normal upstream update also reconciles every ref. This
event-driven setup avoids inactivity-disabled schedules. No routine manual
sync is required, but future policy, repository, or credential changes can
still need administrator attention.
