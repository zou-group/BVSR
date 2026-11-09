# SimpleTES (third-party)

The SimpleTES experiments (paper Sec. 5) run the SimpleTES AI-scientist framework:

| | |
|---|---|
| upstream | https://github.com/wq-will/SimpleTES |
| commit | `a19a54b109db6185ab1f13dd59dd150074b24136` |
| license | GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later) |

SimpleTES is **not** redistributed in this repository. The SimpleTES setup step of `bash reproduce.sh --rerun`
clones the upstream commit and applies `simpletes_a19a54b.patch`. The patch is a modification of
SimpleTES and is distributed under the AGPL-3.0-or-later, the license of the files it modifies.

The two astrodynamics tasks take their task files and evaluators from a second, unpatched upstream commit,
`6f5091d3fd1c6c36b4a35a825d3a3bdfe1cb75cc`.

## Applying it by hand

```bash
git clone https://github.com/wq-will/SimpleTES.git SimpleTES
git -C SimpleTES checkout a19a54b109db6185ab1f13dd59dd150074b24136
git -C SimpleTES apply --check third_party/simpletes/simpletes_a19a54b.patch
git -C SimpleTES apply third_party/simpletes/simpletes_a19a54b.patch
pip install -e SimpleTES
```
