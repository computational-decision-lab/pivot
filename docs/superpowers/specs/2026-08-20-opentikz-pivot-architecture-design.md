# OpenTikZ PIVOT Architecture Design

## Objective

Deploy a pinned, project-local OpenTikZ checkout and use its
`system-block-diagram` template contract to produce an editable, standalone
TikZ architecture figure for the ICLR 2027 paper. The figure explains PIVOT's
decision-preservation mechanism without changing the paper's title, thesis, or
evidence boundary.

## Selected Design

The architecture is a two-row feedback loop. The top row shows the incumbent,
self-improvement operator, candidate transitions, and cheap verifier. The
bottom row is a dashed PIVOT subsystem containing a decision-change score,
paired high-fidelity intervention, differential correction, and corrected
promotion. A bypass explicitly shows that unqueried candidates retain their
proxy/model estimate, while a feedback edge returns the promoted policy to the
next round.

The source is adapted in OpenTikZ Mode A from `system-block-diagram`: the
upstream checkout remains read-only, the project copy is standalone-compilable,
node names remain semantic, geometry is parameterized, and all colors use the
five OpenTikZ palette names. The output targets the ICLR single-column text
width and uses vector PDF in the paper plus SVG for review.

## Deployment And Provenance

`configs/tooling/opentikz.json` pins the repository URL, commit, source hashes,
template ID, and license identifiers. `scripts/bootstrap_opentikz.py` installs
that exact detached commit into ignored `.tools/opentikz` and refuses to mutate
an existing checkout at another revision. The checked-in figure metadata names
the source template and preserves its edit contract.

## Paper Integration

The architecture becomes Figure 3 after the reversal scatter and phase diagram;
the value-versus-improvement contrast consequently becomes Figure 4. The
observer/actor/strategic diagnostic moves to the appendix to preserve the
nine-page main-text limit. The frozen snapshot includes the TikZ source,
metadata, vector PDF, and SVG with hashes.

## Verification

Automated checks validate the lock, bootstrap behavior, OpenTikZ palette and
node contract, standalone compilation, paper placement, snapshot inclusion,
anonymity, private-path exclusion, PDF page limit, and supplement integrity.
The final report remains `CONDITIONAL GO`; adding a method diagram does not
close external scientific or author-side submission gates.

## Assumptions

- Light Okabe-Ito palette.
- ICLR single-column width.
- OpenTikZ upstream commit `359befbf8e8af7ce08e7e387b2c2a198e0ca735d`.
- No OpenTikZ server is required for paper builds; the deployment is a pinned
  local library/tool checkout.
