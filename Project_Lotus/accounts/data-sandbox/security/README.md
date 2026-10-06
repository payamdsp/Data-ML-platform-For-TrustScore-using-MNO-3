# Project Lotus, data-sandbox, security layer

See the [repository README](../../../README.md) for the layer split, the pipeline and the
conventions.

Empty of managed resources. It holds one data source, the permissions boundary lookup.

## Belongs here

KMS keys, aliases and key policies. Secrets Manager containers and their policies. All IAM roles and
policies, including the execution roles the infra layer's services run as. S3 bucket policies.

## The permissions boundary

Every role this root creates must set `permissions_boundary = local.permissions_boundary`, which
`data.tf` and `locals.tf` resolve from the account's contract parameter.

```hcl
resource "aws_iam_role" "example" {
  name                 = "${var.project}-${var.environment}-example"
  permissions_boundary = local.permissions_boundary
}
```

### Naming

| | This account |
| -- | -- |
| Policy | `sandbox-boundary` |
| Published at | `/contract/_account/iam/sandbox-boundary-arn` |

Every other account names its boundary for the account slug, `<slug>-boundary`, published at
`/contract/_account/iam/<slug>-boundary-arn`. Read the target account's real parameter on promotion.

## IAM

Least privilege applies. Scope every policy to the actions and resources the role requires. Do not
attach broad AWS managed policies such as `PowerUserAccess`, and do not grant `Action: "*"` or
`Resource: "*"`.
