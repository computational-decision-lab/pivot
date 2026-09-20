# Claim Audit

The claim registry, permitted evidence, forbidden inferences, and paper-facing scope are checked together.

```json
{
  "body_version_tokens": [],
  "claim_ids": [
    "C1",
    "C2",
    "C3",
    "C4",
    "C5"
  ],
  "claims_have_statements": true,
  "forbidden_inference_count": 5,
  "invalid_terminal_states": {},
  "missing_fields": {},
  "registry": "research/claims_v15.yaml",
  "required_fields": [
    "claim_id",
    "statement",
    "required_experiment",
    "required_terminal_state",
    "allowed_scope",
    "forbidden_scope",
    "paper_location"
  ],
  "required_ids_present": true,
  "statuses": [
    "boundary",
    "conditional",
    "registered"
  ],
  "unique_claim_ids": true,
  "valid": true
}
```
