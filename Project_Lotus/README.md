# Project Lotus

| Directory | What it is |
| -- | -- |
| `project_lotus/` | The Python data and ML project. See [its README](project_lotus/README.md) |
| `accounts/` | The Terraform for the AWS infrastructure it runs on. This file covers that |

Lotus runs in `data-sandbox` (`162591926854`) and nowhere else.

## Data governance

**Confidential Data and Restricted Data are prohibited in this account. Synthetic data only.**

| Classification | Definition |
| -- | -- |
| **Restricted** | Personally identifiable information |
| **Confidential** | Partner data |
| Internal | EnStream intellectual property, such as code and documentation |

Restricted fields in the Exchange schema: `first_name`, `last_name`, `street_number`, `street_name`,
`unit_number`, `city`, `province`, `postal_code`, `date_of_birth`, `phone_number`, `imei`.
`source_institution` is Confidential.

Internal data is permitted. Raise a ticket for any classification question.

## The infra and security split

`accounts/data-sandbox/` holds two root modules. Resource type determines which one a definition
belongs in.

| Directory | Define here |
| -- | -- |
| `infra/` | S3, S3 Tables, Glue, EMR, Athena, DynamoDB, Step Functions, EventBridge, log groups, VPC resources |
| `security/` | KMS keys, aliases and key policies; Secrets Manager; **all IAM roles and policies**; S3 bucket policies |

The split is enforced by the CI roles' permissions. A misplaced definition fails the run.

An authorization error on `role/gha-lotus-data-sandbox-infra` means the resource must be defined in
`security/`.

A `couldn't find resource` error on a data source means the `security/` root has not been applied;
define the resource there and merge that pull request first.

## Pipeline

Triggered by changes under `accounts/data-sandbox/**`.

| Event | Runs | AWS credentials |
| -- | -- | -- |
| Pull request | `fmt -check`, `init -backend=false`, `validate` | none |
| Merge to `main` | `plan` then `apply` | the layer's CI role |

Pull requests do not plan. Plan locally against your own SSO session. Apply runs only on `main`.

Do not add a workflow that assumes the CI roles. Role trust pins `job_workflow_ref` to the shared
workflow, so a workflow defined in this repository receives a token STS rejects.

## EMR capacity and job lifecycle

The Terraform-managed test cluster and the daily Step Functions cluster use
the same instance variables, Spark/YARN configuration, storage sizes, and idle policy.
Gold reuses the daily cluster created by Silver; it does not create another cluster.

| Fleet | Nodes | Instance | vCPUs per node | Memory per node | Purchase option |
| -- | -- | -- | -- | -- | -- |
| Primary | 1 | `m5.xlarge` | 4 | 16 GiB | On-Demand |
| Core | 2 | `r5.2xlarge` | 8 | 64 GiB | On-Demand |
| Task | 2 | `r5.2xlarge` | 8 | 64 GiB | On-Demand |

Workers provide **32 vCPUs and 256 GiB** in total. Instance specifications:
[M5](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html) and
[R5](https://docs.aws.amazon.com/ec2/latest/instancetypes/mo.html).
The primary has a 50 GiB gp3 volume; each worker has a 200 GiB gp3 volume.

Core nodes provide stable driver capacity. Task nodes also use On-Demand so
Spot interruptions do not change capacity or invalidate the Gold pressure test.
Spot task capacity defaults to zero; it can be enabled after establishing a
stable baseline. There is no managed scaling policy to remove workers during a job.

EMR uses `emr-7.14.0` and step concurrency **1**. In YARN cluster deploy mode,
node labels constrain the Spark driver/ApplicationMaster to the core fleet,
not the primary. Gold's configured driver heap is `24g` with `4g` overhead.
Automated arguments come from the live S3 job config, which Terraform seeds
once; subsequent seed-file edits do not update that live object automatically.

Gold retries a failed stage once, then terminates the daily cluster before
sending the failure notification. Successful Gold publish also terminates it.
Silver submits at most two job attempts, with a fresh run ID for its retry;
terminal failure terminates the shared daily cluster and prevents Gold release.
This also affects any other Silver work queued on that cluster.

Both cluster creation paths default to **3,600 seconds of idle time** before
auto-termination. This is a safety net, not a one-hour limit on active jobs;
see [EMR idle criteria](https://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-auto-termination-policy.html).
Independent Silver arrivals more than an hour apart may outlive the idle
cluster and require deliberate recovery. Manual EMR steps do not execute
Step Functions retries or cleanup. Existing executions and running clusters
do not automatically adopt these defaults.

See the [pipeline run handbook](project_lotus/docs/docs/transient_emr_and_bronze_arrivals.md)
for artifact checks, live configuration, and manual test submission.

## Raising a ticket

If you run into an issue, raise a ticket on the Project Lotus Linear team and tag InfraSec:
<https://linear.app/enstream-workspace/team/LOTUS/overview>

## Conventions

Each requirement below supports a control elsewhere in the system.

- **Prefix every resource name with `var.project`** (`lotus`). IAM and KMS policies are written
  against `lotus-*`; an unprefixed resource falls outside them silently.
- **Build names from `var.environment`**, never a literal `sandbox`. Promotion is a directory copy
  with that one variable changed.
- **No hardcoded account ids or ARNs.** Resolve by name, or read `/contract/_account/*` from SSM.
- **No account names in file banners or comments.** A promotion copies them and they become false.
- `required_version = "~> 1.15.0"`, provider pinned exactly, `.terraform.lock.hcl` committed.
- Run `terraform fmt -recursive` before committing. CI fails on unformatted files.
- **Never commit** state, `.terraform/`, saved plans, or `.tfvars` with real values.
- Branch and open a PR; never push `main`. Keep infra and security changes in separate PRs.
